import os
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from sklearn.preprocessing import StandardScaler, OneHotEncoder, normalize
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

# =========================
# PAGE CONFIG & BASIC THEME
# =========================
st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👗",
    layout="wide",
)

# Simple CSS for cards and section titles
st.markdown(
    """
    <style>
    .main {
        background-color: #fafafa;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .subsection-title {
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 1.0rem;
        margin-bottom: 0.25rem;
    }
    .product-card {
        padding: 0.8rem;
        background-color: white;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# CONFIG: update these paths
# =========================
STYLES_CSV_PATH = "Dataset/FeatureEngineered_StyleData.csv"

IMG_EMB_NPY_PATH = "Dataset/image_embeddings_clip.npy"
IMG_EMB_IDS_CSV_PATH = "Dataset/image_embeddings_ids.csv"

SIAMESE_EMB_NPY_PATH = "Dataset/siamese_embeddings.npy"
SIAMESE_IDS_NPY_PATH = "Dataset/siamese_ids.npy"
SIAMESE_ENCODER_PT_PATH = "Dataset/siamese_encoder.pt"

IMAGES_DIR = "fashion-dataset/fashion-dataset/images_small"   # e.g. images_small/10000.jpg


# =========================
# 1. Data loading helpers
# =========================
@st.cache_data
def load_styles():
    df = pd.read_csv(STYLES_CSV_PATH)
    df["id"] = df["id"].astype(int)
    return df


@st.cache_data
def load_image_embeddings():
    embs = np.load(IMG_EMB_NPY_PATH)  # (N, 512)
    ids = pd.read_csv(IMG_EMB_IDS_CSV_PATH)["id"].astype(int).values
    return embs, ids


def align_embeddings(styles, img_embs, img_ids):
    """Align CLIP image embeddings with styles df by id."""
    common_ids = np.intersect1d(styles["id"].values, img_ids)
    styles_aligned = styles[styles["id"].isin(common_ids)].copy()
    styles_aligned["id"] = styles_aligned["id"].astype(int)

    id_to_idx = {int(i): idx for idx, i in enumerate(img_ids)}
    emb_list = [img_embs[id_to_idx[i]] for i in styles_aligned["id"].values]
    img_embs_aligned = np.stack(emb_list)

    return styles_aligned, img_embs_aligned


# =========================
# 2. Meta + text features
# =========================
@st.cache_resource
def build_meta_text_pipeline(styles: pd.DataFrame):
    # Helper to safely get a text column in lowercase
    def safe_text(col_name, df=styles):
        if col_name in df.columns:
            return df[col_name].fillna("").astype(str).str.lower()
        else:
            # returns empty Series with same index
            return pd.Series([""] * len(df), index=df.index)

    # Build style_signature if not present, in a normalized way
    if "style_signature" not in styles.columns:
        styles["style_signature"] = (
            safe_text("gender") + " " +
            safe_text("subCategory") + " " +
            safe_text("articleType") + " " +
            safe_text("baseColour") + " " +
            safe_text("usage") + " " +
            (safe_text("season_group") if "season_group" in styles.columns else safe_text("season"))
        )

    num_cols = [
        "log_effective_price", "discount_ratio", "myntraRating",
        "vat", "num_style_options", "total_inventoryCount",
        "category_strength", "isEMIEnabled", "isFragile", "isTryAndBuyEnabled",
        "isHazmat", "isJewellery", "isReturnable", "isExchangeable",
        "pickupEnabled", "isLarge", "codEnabled", "any_in_stock",
        "has_image", "is_discounted"
    ]
    num_cols = [c for c in num_cols if c in styles.columns]

    cat_cols = [
        "brandName", "ageGroup", "gender", "baseColour", "colour1",
        "fashionType", "season_group" if "season_group" in styles.columns else "season",
        "usage", "masterCategory", "subCategory", "articleType",
        "articleAttr_Pattern", "articleAttr_Body_or_Garment_Size",
        "articleAttr_Fit", "articleAttr_Sleeve_Length", "articleAttr_Neck",
        "articleAttr_Fabric", "articleAttr_Occasion", "articleAttr_Type",
        "displayCat_primary", "price_bucket", "era", "segment"
    ]
    cat_cols = [c for c in cat_cols if c in styles.columns]

    text_col = "style_signature"

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    meta_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    X_meta = meta_preprocessor.fit_transform(styles)

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_text = tfidf.fit_transform(styles[text_col].fillna(""))

    svd = TruncatedSVD(n_components=64, random_state=42)
    X_text_red = svd.fit_transform(X_text)   # dense
    X_text_red = csr_matrix(X_text_red)      # sparse

    X_meta_text = hstack([X_meta, X_text_red])

    return X_meta_text, meta_preprocessor, tfidf, svd, num_cols, cat_cols, text_col


def build_query_meta_vector(
    user_meta: dict,
    meta_preprocessor,
    tfidf,
    svd,
    num_cols,
    cat_cols,
    text_col: str
):
    df_q = pd.DataFrame([user_meta])

    # Fill missing numeric and categorical columns
    for c in num_cols:
        if c not in df_q.columns:
            df_q[c] = 0
    for c in cat_cols:
        if c not in df_q.columns:
            df_q[c] = "Unknown"

    # Build style_signature the SAME WAY as in training
    if text_col not in df_q.columns:
        def safe_text_q(col_name):
            if col_name in df_q.columns:
                return df_q[col_name].fillna("").astype(str).str.lower()
            else:
                return pd.Series([""], index=df_q.index)

        df_q[text_col] = (
            safe_text_q("gender") + " " +
            safe_text_q("subCategory") + " " +
            safe_text_q("articleType") + " " +
            safe_text_q("baseColour") + " " +
            safe_text_q("usage") + " " +
            (safe_text_q("season_group") if "season_group" in df_q.columns else safe_text_q("season"))
        )

    # Transform with the SAME fitted objects
    X_meta_q = meta_preprocessor.transform(df_q)
    X_text_q = tfidf.transform(df_q[text_col].fillna(""))
    X_text_q_red = svd.transform(X_text_q)
    X_text_q_red = csr_matrix(X_text_q_red)

    X_q = hstack([X_meta_q, X_text_q_red])
    return X_q


# =========================
# 3. CLIP model
# =========================
@st.cache_resource
def load_clip_model():
    MODEL_NAME = "ViT-B-32"
    PRETRAINED = "laion2b_s34b_b79k"
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, preprocess, device


def encode_uploaded_image(file_obj, model, preprocess, device):
    img = Image.open(file_obj).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(img_t)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return img, feat.cpu().numpy().reshape(-1)   # (512,)


# =========================
# 4. Siamese encoder + embeddings
# =========================
class ProductEncoder(nn.Module):
    def __init__(self, dim_img=512, dim_meta=None, hidden_dim=512, out_dim=128):
        super().__init__()
        assert dim_meta is not None, "dim_meta must be provided"

        in_dim = dim_img + dim_meta
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, img_emb, meta_emb):
        x = torch.cat([img_emb, meta_emb], dim=1)
        z = self.net(x)
        z = F.normalize(z, p=2, dim=1)  # L2-normalize embeddings
        return z


@st.cache_data
def load_siamese_embeddings():
    style_embs = np.load(SIAMESE_EMB_NPY_PATH)          # (N, emb_dim)
    style_ids = np.load(SIAMESE_IDS_NPY_PATH).astype(int)
    id_to_row = {int(pid): idx for idx, pid in enumerate(style_ids)}
    return style_embs, style_ids, id_to_row


@st.cache_resource
def load_siamese_encoder(meta_dim: int, emb_dim: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ProductEncoder(
        dim_img=512,
        dim_meta=meta_dim,
        hidden_dim=512,
        out_dim=emb_dim,
    )
    state = torch.load(SIAMESE_ENCODER_PT_PATH, map_location=device)
    # adapt here if your checkpoint is wrapped, e.g. state["encoder"]
    encoder.load_state_dict(state)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder, device


# =========================
# 5. Recommenders
# =========================
def get_nearest_by_image(query_emb, img_embs_norm, styles, topn=10):
    sims = cosine_similarity(query_emb.reshape(1, -1), img_embs_norm).ravel()
    order = sims.argsort()[::-1][:topn]
    recs = styles.iloc[order].copy()
    recs["similarity_img_query"] = sims[order]
    return recs


def recommend_by_meta_query(X_query_meta, styles, X_meta_text, topn=10):
    sims = cosine_similarity(X_query_meta, X_meta_text).ravel()
    order = sims.argsort()[::-1][:topn]
    recs = styles.iloc[order].copy()
    recs["similarity_meta_query"] = sims[order]
    return recs


def recommend_hybrid_query(
    img_query_emb,
    X_query_meta,
    styles,
    img_embs_norm,
    X_meta_text,
    alpha=0.6,
    topn=10
):
    img_sim = cosine_similarity(img_query_emb.reshape(1, -1), img_embs_norm).ravel()
    meta_sim = cosine_similarity(X_query_meta, X_meta_text).ravel()
    sims = alpha * img_sim + (1 - alpha) * meta_sim

    order = sims.argsort()[::-1][:topn]
    recs = styles.iloc[order].copy()
    recs["similarity_img_query"] = img_sim[order]
    recs["similarity_meta_query"] = meta_sim[order]
    recs["similarity_hybrid_query"] = sims[order]
    return recs


def recommend_siamese_catalog(product_id, styles, style_embs, id_to_row, topn=10):
    if product_id not in id_to_row:
        raise ValueError(f"Unknown product id {product_id}")

    i = id_to_row[product_id]
    query_vec = style_embs[i].reshape(1, -1)
    sims = cosine_similarity(query_vec, style_embs).ravel()

    order = sims.argsort()[::-1]
    order = [idx for idx in order if idx != i][:topn]

    recs = styles.iloc[order].copy()
    recs["similarity_siamese"] = sims[order]
    return recs


def recommend_siamese_from_query(
    query_img_emb: np.ndarray,
    X_query_meta,
    styles: pd.DataFrame,
    style_embs: np.ndarray,
    encoder: nn.Module,
    device: str,
    topn: int = 10
):
    # meta to dense
    meta_dense = X_query_meta.toarray().astype(np.float32)  # (1, meta_dim)
    img_np = query_img_emb.astype(np.float32).reshape(1, -1)

    img_t = torch.from_numpy(img_np).to(device)
    meta_t = torch.from_numpy(meta_dense).to(device)

    with torch.no_grad():
        z_q = encoder(img_t, meta_t)        # (1, emb_dim)
        z_q = z_q.cpu().numpy()

    sims = cosine_similarity(z_q, style_embs).ravel()
    order = sims.argsort()[::-1][:topn]

    recs = styles.iloc[order].copy()
    recs["similarity_siamese"] = sims[order]
    return recs


# =========================
# 6. UI helpers
# =========================
def get_image_path(pid):
    return os.path.join(IMAGES_DIR, f"{int(pid)}.jpg")


def filter_recs_with_images(recs: pd.DataFrame) -> pd.DataFrame:
    keep_idxs = []
    for idx, row in recs.iterrows():
        img_path = get_image_path(row["id"])
        if os.path.exists(img_path):
            keep_idxs.append(idx)
    return recs.loc[keep_idxs].reset_index(drop=True)


def show_product_card(row, title=None):
    pid = int(row["id"])
    img_path = get_image_path(pid)

    if title:
        st.markdown(f"<div class='subsection-title'>{title}</div>", unsafe_allow_html=True)

    with st.container():
        cols = st.columns([1, 2])
        with cols[0]:
            if os.path.exists(img_path):
                st.image(img_path, caption=f"ID: {pid}", width=180)
            else:
                st.write(f"[No image found for {pid}]")

        with cols[1]:
            st.markdown("<div class='product-card'>", unsafe_allow_html=True)
            st.markdown(f"**ID:** {pid}")
            if "brandName" in row:
                st.markdown(f"**Brand:** {row['brandName']}")
            if "articleType" in row:
                st.markdown(f"**Type:** {row['articleType']}")
            if "baseColour" in row:
                st.markdown(f"**Colour:** {row['baseColour']}")
            if "effective_price" in row:
                st.markdown(f"**Price:** {row['effective_price']}")
            if "price_bucket" in row:
                st.markdown(f"**Price Bucket:** {row['price_bucket']}")
            if "season_group" in row:
                st.markdown(f"**Season:** {row['season_group']}")
            elif "season" in row:
                st.markdown(f"**Season:** {row['season']}")
            if "usage" in row:
                st.markdown(f"**Usage:** {row['usage']}")
            st.markdown("</div>", unsafe_allow_html=True)


def collect_user_metadata(styles: pd.DataFrame):
    # Options from catalog
    gender_options = (
        sorted(styles["gender"].dropna().unique().tolist())
        if "gender" in styles.columns else ["Men", "Women", "Unisex"]
    )
    age_group_options = (
        sorted(styles["ageGroup"].dropna().unique().tolist())
        if "ageGroup" in styles.columns else []
    )
    master_cat_options = (
        sorted(styles["masterCategory"].dropna().unique().tolist())
        if "masterCategory" in styles.columns else ["Apparel"]
    )
    base_colour_options = (
        sorted(styles["baseColour"].dropna().unique().tolist())
        if "baseColour" in styles.columns else []
    )
    season_col = "season_group" if "season_group" in styles.columns else "season"
    season_options = (
        sorted(styles[season_col].dropna().unique().tolist())
        if season_col in styles.columns else []
    )
    usage_options = (
        sorted(styles["usage"].dropna().unique().tolist())
        if "usage" in styles.columns else []
    )

    # First row of inputs
    col1, col2 = st.columns(2)
    with col1:
        q_gender = st.selectbox("Gender", gender_options)
        q_age = st.selectbox("Age Group", age_group_options) if age_group_options else "Adults"
        q_master = st.selectbox("Master Category", master_cat_options)
    with col2:
        q_colour = st.selectbox("Base Colour", base_colour_options)
        q_season = st.selectbox("Season / Season Group", season_options)
        q_usage = st.selectbox("Usage", usage_options)

    # SubCategory options (filtered by master category if possible)
    if "masterCategory" in styles.columns and "subCategory" in styles.columns:
        subcat_options = sorted(
            styles.loc[styles["masterCategory"] == q_master, "subCategory"]
            .dropna()
            .unique()
            .tolist()
        )
        if not subcat_options:
            subcat_options = sorted(styles["subCategory"].dropna().unique().tolist())
    else:
        subcat_options = sorted(styles["subCategory"].dropna().unique().tolist()) if "subCategory" in styles.columns else []

    q_subcat = st.selectbox("Sub Category", subcat_options) if subcat_options else ""

    # ArticleType options (optionally filtered by subCategory)
    if "articleType" in styles.columns:
        if q_subcat:
            article_type_options = sorted(
                styles.loc[styles["subCategory"] == q_subcat, "articleType"]
                .dropna()
                .unique()
                .tolist()
            )
            if not article_type_options:
                article_type_options = sorted(styles["articleType"].dropna().unique().tolist())
        else:
            article_type_options = sorted(styles["articleType"].dropna().unique().tolist())
    else:
        article_type_options = []
    q_article = st.selectbox("Article Type", article_type_options) if article_type_options else ""

    # Approx price input
    q_price = st.number_input("Approx Price", min_value=0.0, value=1000.0, step=50.0)

    def bucket_price(p):
        if p < 1000:
            return "low"
        elif p < 3000:
            return "mid"
        else:
            return "high"

    user_meta = {
        "gender": q_gender,
        "ageGroup": q_age,
        "masterCategory": q_master,
        "subCategory": q_subcat,
        "articleType": q_article,
        "baseColour": q_colour,
        "usage": q_usage,
        "effective_price": q_price,
        "log_effective_price": np.log(q_price + 1),
        season_col: q_season,
        "price_bucket": bucket_price(q_price),
    }

    # Optional: reconstruct segment if used in training
    if "segment" in styles.columns:
        user_meta["segment"] = f"{q_gender}_{q_age}"

    return user_meta


# =========================
# 7. Simple evaluation (Compare Models tab)
# =========================
def evaluate_models_for_query(
    query_id: int,
    styles: pd.DataFrame,
    img_embs_norm: np.ndarray,
    X_meta_text,
    style_embs: np.ndarray,
    siamese_id_to_row: dict,
    siamese_encoder: nn.Module,
    siamese_device: str,
    alpha: float = 0.6,
    topn_eval: int = 10,
):
    id_to_idx = {pid: idx for idx, pid in enumerate(styles["id"].values)}
    if query_id not in id_to_idx:
        raise ValueError("Query id not found in styles")

    i = id_to_idx[query_id]
    q_row = styles.iloc[i]
    q_price = q_row.get("effective_price", np.nan)

    q_img = img_embs_norm[i]
    X_q_meta = X_meta_text[i]

    metrics_list = []

    def _compute_metrics(name, recs_df):
        if recs_df is None or recs_df.empty:
            return

        recs = recs_df[recs_df["id"] != query_id].copy()
        if recs.empty:
            return

        cat_match = None
        if "articleType" in recs and pd.notna(q_row.get("articleType", np.nan)):
            cat_match = (recs["articleType"] == q_row["articleType"]).mean()

        subcat_match = None
        if "subCategory" in recs and pd.notna(q_row.get("subCategory", np.nan)):
            subcat_match = (recs["subCategory"] == q_row["subCategory"]).mean()

        brand_match = None
        if "brandName" in recs and pd.notna(q_row.get("brandName", np.nan)):
            brand_match = (recs["brandName"] == q_row["brandName"]).mean()

        price_drift = None
        if "effective_price" in recs and not pd.isna(q_price):
            price_drift = (recs["effective_price"] - q_price).abs().mean()

        metrics_list.append(
            {
                "model": name,
                "category_match": cat_match,
                "subcategory_match": subcat_match,
                "brand_match": brand_match,
                "avg_price_drift": price_drift,
            }
        )

    # image
    recs_img = get_nearest_by_image(q_img, img_embs_norm, styles, topn=topn_eval + 1)
    _compute_metrics("image", recs_img)

    # metadata
    recs_meta = recommend_by_meta_query(X_q_meta, styles, X_meta_text, topn=topn_eval + 1)
    _compute_metrics("metadata", recs_meta)

    # hybrid
    recs_hybrid = recommend_hybrid_query(
        q_img,
        X_q_meta,
        styles,
        img_embs_norm,
        X_meta_text,
        alpha=alpha,
        topn=topn_eval + 1,
    )
    _compute_metrics("hybrid", recs_hybrid)

    # siamese (catalog)
    recs_siamese = None
    if query_id in siamese_id_to_row:
        recs_siamese = recommend_siamese_catalog(
            query_id, styles, style_embs, siamese_id_to_row, topn=topn_eval
        )
        _compute_metrics("siamese", recs_siamese)

    return pd.DataFrame(metrics_list)


# =========================
# 8. Main app
# =========================
def main():
    st.markdown(
        """
        <h2>👗 Fashion Product Recommender</h2>
        <p>
        This demo shows different recommendation strategies on a fashion catalog:
        <b>image similarity</b> (CLIP), <b>metadata similarity</b>, a <b>hybrid</b> of both,
        and a <b>deep Siamese encoder</b> that learns a joint similarity space.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Load base data & models
    with st.spinner("Loading data and models..."):
        styles_raw = load_styles()
        img_embs, img_ids = load_image_embeddings()
        styles, img_embs_aligned = align_embeddings(styles_raw, img_embs, img_ids)
        img_embs_norm = normalize(img_embs_aligned)

        X_meta_text, meta_preprocessor, tfidf, svd, num_cols, cat_cols, text_col = build_meta_text_pipeline(styles)
        meta_dim = X_meta_text.shape[1]

        model, preprocess, clip_device = load_clip_model()

        style_embs, siamese_ids, siamese_id_to_row = load_siamese_embeddings()
        siamese_encoder, siamese_device = load_siamese_encoder(
            meta_dim=meta_dim,
            emb_dim=style_embs.shape[1],
        )

    # Sidebar global controls
    st.sidebar.header("Global Settings")
    topn = st.sidebar.slider("Top-N recommendations", 5, 20, 10, 1)
    alpha_hybrid = st.sidebar.slider("Hybrid weight α (image vs metadata)", 0.0, 1.0, 0.6, 0.05)

    # Tabs: demo vs comparison
    tab_demo, tab_compare = st.tabs(["🔍 Recommender Demo", "📊 Compare Models"])

    # ------------------ TAB 1: DEMO ------------------
    with tab_demo:
        st.markdown("<div class='section-title'>Choose a recommendation mode</div>", unsafe_allow_html=True)

        mode = st.radio(
            "Mode",
            ["Image only", "Metadata only", "Hybrid (Image + Metadata)", "Siamese"],
            horizontal=True,
        )

        # Common: metadata form (for modes that need it)
        user_meta = None
        if mode in ["Metadata only", "Hybrid (Image + Metadata)", "Siamese"]:
            st.markdown("<div class='subsection-title'>Describe your item (metadata)</div>", unsafe_allow_html=True)
            user_meta = collect_user_metadata(styles)

        # Common: image upload (for modes that need it)
        uploaded_file_image = None
        if mode in ["Image only", "Hybrid (Image + Metadata)"]:
            uploaded_file_image = st.file_uploader(
                "Upload a clothing image (JPG/PNG)",
                type=["jpg", "jpeg", "png"],
                key="main_uploader"
            )

        # Siamese query type
        siamese_query_type = None
        uploaded_file_siamese = None
        selected_pid = None
        if mode == "Siamese":
            st.markdown("<div class='subsection-title'>Siamese query type</div>", unsafe_allow_html=True)
            siamese_query_type = st.radio(
                "How do you want to query?",
                ["Use existing catalog item", "Upload image + metadata"],
                horizontal=True,
            )

            if siamese_query_type == "Use existing catalog item":
                sample_ids = sorted(styles["id"].sample(min(500, len(styles)), random_state=42).tolist())
                selected_pid = st.selectbox("Choose product ID from catalog", sample_ids)
                if st.button("Show query product", key="show_siamese_query"):
                    q_row = styles[styles["id"] == selected_pid].iloc[0]
                    st.markdown("<div class='subsection-title'>Query Product (Siamese)</div>", unsafe_allow_html=True)
                    show_product_card(q_row)
            else:
                uploaded_file_siamese = st.file_uploader(
                    "Upload image for Siamese similarity (JPG/PNG)",
                    type=["jpg", "jpeg", "png"],
                    key="siamese_uploader"
                )

        # Action button
        if st.button("🚀 Generate Recommendations", key="generate_button"):

            # IMAGE ONLY
            if mode == "Image only":
                if uploaded_file_image is None:
                    st.error("Please upload an image.")
                else:
                    query_img, query_emb = encode_uploaded_image(uploaded_file_image, model, preprocess, clip_device)
                    st.markdown("<div class='subsection-title'>Uploaded Image</div>", unsafe_allow_html=True)
                    st.image(query_img, caption="Query Image", width=260)

                    st.markdown("<div class='subsection-title'>Image-based Recommendations</div>", unsafe_allow_html=True)
                    recs_img = get_nearest_by_image(query_emb, img_embs_norm, styles, topn=topn)
                    recs_img = filter_recs_with_images(recs_img)
                    for _, row in recs_img.iterrows():
                        show_product_card(row)

            # METADATA ONLY
            elif mode == "Metadata only":
                X_query_meta = build_query_meta_vector(
                    user_meta,
                    meta_preprocessor,
                    tfidf,
                    svd,
                    num_cols,
                    cat_cols,
                    text_col
                )
                st.markdown("<div class='subsection-title'>Metadata/Text-based Recommendations</div>", unsafe_allow_html=True)
                recs_meta = recommend_by_meta_query(X_query_meta, styles, X_meta_text, topn=topn)
                recs_meta = filter_recs_with_images(recs_meta)
                for _, row in recs_meta.iterrows():
                    show_product_card(row)

            # HYBRID
            elif mode == "Hybrid (Image + Metadata)":
                if uploaded_file_image is None:
                    st.error("Please upload an image.")
                else:
                    query_img, query_emb = encode_uploaded_image(uploaded_file_image, model, preprocess, clip_device)
                    st.markdown("<div class='subsection-title'>Uploaded Image</div>", unsafe_allow_html=True)
                    st.image(query_img, caption="Query Image", width=260)

                    X_query_meta = build_query_meta_vector(
                        user_meta,
                        meta_preprocessor,
                        tfidf,
                        svd,
                        num_cols,
                        cat_cols,
                        text_col
                    )

                    # st.markdown("<div class='subsection-title'>Image-based Recommendations</div>", unsafe_allow_html=True)
                    # recs_img = get_nearest_by_image(query_emb, img_embs_norm, styles, topn=topn)
                    # recs_img = filter_recs_with_images(recs_img)
                    # for _, row in recs_img.iterrows():
                    #     show_product_card(row)

                    # st.markdown("<div class='subsection-title'>Metadata/Text-based Recommendations</div>", unsafe_allow_html=True)
                    # recs_meta = recommend_by_meta_query(X_query_meta, styles, X_meta_text, topn=topn)
                    # recs_meta = filter_recs_with_images(recs_meta)
                    # for _, row in recs_meta.iterrows():
                    #     show_product_card(row)

                    st.markdown("<div class='subsection-title'>Hybrid Recommendations (image + metadata)</div>", unsafe_allow_html=True)
                    recs_hybrid = recommend_hybrid_query(
                        query_emb,
                        X_query_meta,
                        styles,
                        img_embs_norm,
                        X_meta_text,
                        alpha=alpha_hybrid,
                        topn=topn
                    )
                    recs_hybrid = filter_recs_with_images(recs_hybrid)
                    for _, row in recs_hybrid.iterrows():
                        show_product_card(row)

            # SIAMESE
            elif mode == "Siamese":
                if siamese_query_type == "Use existing catalog item":
                    if selected_pid is None:
                        st.error("Please select a product ID.")
                    elif selected_pid not in siamese_id_to_row:
                        st.error("Selected ID does not have a Siamese embedding.")
                    else:
                        q_row = styles[styles["id"] == selected_pid].iloc[0]
                        st.markdown("<div class='subsection-title'>Query Product (Siamese)</div>", unsafe_allow_html=True)
                        show_product_card(q_row)

                        st.markdown("<div class='subsection-title'>Siamese-based Recommendations</div>", unsafe_allow_html=True)
                        recs_siamese = recommend_siamese_catalog(
                            selected_pid,
                            styles,
                            style_embs,
                            siamese_id_to_row,
                            topn=topn
                        )
                        recs_siamese = filter_recs_with_images(recs_siamese)
                        for _, row in recs_siamese.iterrows():
                            show_product_card(row)

                else:  # Upload image + metadata
                    if uploaded_file_siamese is None:
                        st.error("Please upload an image for Siamese query.")
                    else:
                        query_img, query_emb = encode_uploaded_image(
                            uploaded_file_siamese,
                            model,
                            preprocess,
                            siamese_device,
                        )
                        st.markdown("<div class='subsection-title'>Uploaded Image (Siamese)</div>", unsafe_allow_html=True)
                        st.image(query_img, caption="Query Image", width=260)

                        X_query_meta = build_query_meta_vector(
                            user_meta,
                            meta_preprocessor,
                            tfidf,
                            svd,
                            num_cols,
                            cat_cols,
                            text_col
                        )

                        st.markdown("<div class='subsection-title'>Siamese-based Recommendations (uploaded image + metadata)</div>", unsafe_allow_html=True)
                        recs_siamese = recommend_siamese_from_query(
                            query_emb,
                            X_query_meta,
                            styles,
                            style_embs,
                            siamese_encoder,
                            siamese_device,
                            topn=topn,
                        )
                        recs_siamese = filter_recs_with_images(recs_siamese)
                        for _, row in recs_siamese.iterrows():
                            show_product_card(row)

    # ------------------ TAB 2: COMPARE MODELS ------------------
    with tab_compare:
        st.markdown("<div class='section-title'>Model Comparison</div>", unsafe_allow_html=True)
        st.write(
            "Pick a product from the catalog and compare how different recommenders behave: "
            "image-based, metadata-based, hybrid, and Siamese."
        )

        sample_ids = sorted(styles["id"].sample(min(500, len(styles)), random_state=0).tolist())
        query_id = st.selectbox("Choose a query product ID", sample_ids, key="compare_query_id")
        topn_eval = st.slider("Top-N to evaluate", 5, 30, 10, 1, key="topn_eval")

        if st.button("🔎 Evaluate Models", key="eval_button"):
            try:
                with st.spinner("Evaluating models on this query..."):
                    df_metrics = evaluate_models_for_query(
                        query_id,
                        styles,
                        img_embs_norm,
                        X_meta_text,
                        style_embs,
                        siamese_id_to_row,
                        siamese_encoder,
                        siamese_device,
                        alpha=alpha_hybrid,
                        topn_eval=topn_eval,
                    )
                if df_metrics.empty:
                    st.warning("No metrics computed (maybe this ID is not suitable?).")
                else:
                    st.markdown("<div class='subsection-title'>Query Product</div>", unsafe_allow_html=True)
                    q_row = styles[styles["id"] == query_id].iloc[0]
                    show_product_card(q_row)

                    st.markdown("<div class='subsection-title'>Metrics Summary</div>", unsafe_allow_html=True)
                    st.dataframe(df_metrics, use_container_width=True)

                    st.markdown("<div class='subsection-title'>Category Match by Model</div>", unsafe_allow_html=True)
                    st.bar_chart(
                        df_metrics.set_index("model")[["category_match", "subcategory_match"]]
                    )

                    st.markdown("<div class='subsection-title'>Average Price Drift by Model</div>", unsafe_allow_html=True)
                    st.bar_chart(
                        df_metrics.set_index("model")[["avg_price_drift"]]
                    )
            except Exception as e:
                st.error(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()
