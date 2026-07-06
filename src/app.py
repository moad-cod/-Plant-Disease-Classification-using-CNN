import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import pickle
import os
import time
import logging
from html import escape

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="PlantGuard AI | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --forest:#0B2E1A;
  --leaf:#22C55E;
  --lime:#A3E635;
  --mint:#ECFDF5;
  --ink:#0F172A;
  --muted:#64748B;
  --line:#D1FAE5;
  --white:#FFFFFF;
  --amber:#D97706;
  --red:#DC2626;
}

*, *::before, *::after { box-sizing:border-box; }
html, body, [class*="css"] {
  font-family:'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color:var(--ink);
  background:
    radial-gradient(circle at 12% 0%, rgba(163,230,53,.24), transparent 28rem),
    radial-gradient(circle at 90% 8%, rgba(34,197,94,.16), transparent 24rem),
    linear-gradient(180deg, #F7FEFA 0%, #ECFDF5 48%, #F8FAFC 100%);
}
#MainMenu, footer, header { visibility:hidden; }
.block-container { max-width:1180px; padding:2rem 1.25rem 3.5rem !important; }
section[data-testid="stSidebar"] { display:none; }
h1, h2, h3, p { letter-spacing:0; }

@keyframes fadeUp {
  from { opacity:0; transform:translateY(18px); }
  to { opacity:1; transform:translateY(0); }
}
@keyframes scaleIn {
  from { opacity:0; transform:scale(.98); }
  to { opacity:1; transform:scale(1); }
}
@keyframes fillBar {
  from { width:0; }
}

.hero {
  position:relative;
  overflow:hidden;
  border:1px solid rgba(209,250,229,.9);
  border-radius:8px;
  padding:2.25rem;
  min-height:310px;
  background:
    linear-gradient(105deg, rgba(11,46,26,.96), rgba(15,92,49,.9) 52%, rgba(34,197,94,.78)),
    url("https://images.unsplash.com/photo-1520412099551-62b6bafeb5bb?auto=format&fit=crop&w=1600&q=80");
  background-size:cover;
  background-position:center;
  box-shadow:0 24px 70px rgba(11,46,26,.18);
  animation:fadeUp .55s ease both;
}
.hero::after {
  content:"";
  position:absolute;
  inset:0;
  background:linear-gradient(90deg, rgba(11,46,26,.92), rgba(11,46,26,.62) 52%, rgba(11,46,26,.18));
}
.hero-content { position:relative; z-index:1; max-width:720px; color:#F3F8F1; }
.eyebrow {
  display:inline-flex;
  align-items:center;
  gap:.5rem;
  padding:.42rem .75rem;
  border:1px solid rgba(163,230,53,.52);
  border-radius:999px;
  background:rgba(7,19,13,.45);
  color:#D9F99D;
  font-size:.78rem;
  font-weight:700;
}
.hero h1 {
  margin:.95rem 0 .35rem;
  font-size:clamp(2.5rem, 6vw, 4.8rem);
  line-height:.95;
  font-weight:800;
  color:#FFFFFF;
}
.hero h2 {
  margin:0 0 .9rem;
  color:#BBF7D0;
  font-size:clamp(1.05rem, 2.2vw, 1.45rem);
  font-weight:700;
}
.hero p {
  max-width:620px;
  margin:0;
  color:#E6F7EA;
  font-size:1rem;
  line-height:1.7;
}
.hero-metrics {
  position:relative;
  z-index:1;
  display:grid;
  grid-template-columns:repeat(4, minmax(0, 1fr));
  gap:.7rem;
  margin-top:1.7rem;
  max-width:760px;
}
.metric {
  border:1px solid rgba(209,250,229,.22);
  border-radius:8px;
  background:rgba(7,19,13,.34);
  padding:.8rem .9rem;
  color:#F3F8F1;
}
.metric strong { display:block; font-size:1.05rem; color:#A3E635; }
.metric span { display:block; margin-top:.16rem; font-size:.74rem; color:#D8F3DC; }

.tabs-wrap { margin-top:1.15rem; }
[data-baseweb="tab-list"] {
  gap:.4rem !important;
  border-bottom:0 !important;
  background:rgba(255,255,255,.72) !important;
  border:1px solid var(--line) !important;
  border-radius:8px !important;
  padding:.35rem !important;
}
[data-baseweb="tab"] {
  border-radius:7px !important;
  color:var(--muted) !important;
  font-weight:700 !important;
  padding:.55rem 1rem !important;
}
[aria-selected="true"] {
  background:var(--forest) !important;
  color:#FFFFFF !important;
}

.section-head {
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap:1rem;
  margin:1.4rem 0 .8rem;
}
.section-head h3 {
  margin:0;
  color:var(--forest);
  font-size:1.35rem;
}
.section-head p {
  margin:.22rem 0 0;
  color:var(--muted);
  line-height:1.6;
}
.card {
  border:1px solid var(--line);
  border-radius:8px;
  background:rgba(255,255,255,.9);
  box-shadow:0 18px 50px rgba(15,23,42,.08);
  padding:1.15rem;
  animation:fadeUp .45s ease both;
}
.upload-card {
  position:relative;
  border:1.5px dashed rgba(34,197,94,.78);
  border-radius:8px;
  background:linear-gradient(180deg, rgba(255,255,255,.96), rgba(236,253,245,.86));
  box-shadow:0 18px 52px rgba(34,197,94,.14);
  padding:1.4rem;
}
.upload-icon {
  width:48px;
  height:48px;
  display:grid;
  place-items:center;
  border-radius:8px;
  background:#DCFCE7;
  color:var(--forest);
  font-size:1.5rem;
  margin-bottom:.8rem;
}
.upload-title { margin:0; color:var(--forest); font-size:1.15rem; font-weight:800; }
.upload-copy { margin:.3rem 0 .9rem; color:var(--muted); line-height:1.55; }
.format-row { display:flex; flex-wrap:wrap; gap:.45rem; margin:.75rem 0 1rem; }
.chip {
  display:inline-flex;
  align-items:center;
  gap:.35rem;
  border:1px solid var(--line);
  border-radius:999px;
  padding:.33rem .62rem;
  background:#FFFFFF;
  color:#14532D;
  font-size:.78rem;
  font-weight:800;
}
[data-testid="stFileUploader"] {
  border:0;
  padding:0;
}
[data-testid="stFileUploader"] section {
  border:1px solid rgba(34,197,94,.35) !important;
  border-radius:8px !important;
  background:#FFFFFF !important;
}
[data-testid="stFileUploader"] button {
  border-radius:7px !important;
  border:1px solid var(--forest) !important;
  color:var(--forest) !important;
  font-weight:800 !important;
}

.status-card {
  border:1px solid var(--line);
  border-radius:8px;
  background:#FFFFFF;
  padding:1rem;
}
.model-pill {
  display:inline-flex;
  align-items:center;
  gap:.35rem;
  padding:.34rem .62rem;
  border-radius:999px;
  background:#F0FDF4;
  border:1px solid var(--line);
  color:#166534;
  font-size:.78rem;
  font-weight:800;
}
.preview-note { color:var(--muted); font-size:.82rem; margin-top:.55rem; }
.stImage img { border-radius:8px; }
.stButton>button {
  min-height:3rem;
  border-radius:8px !important;
  border:0 !important;
  background:linear-gradient(135deg, var(--forest), #15803D) !important;
  color:#FFFFFF !important;
  font-weight:800 !important;
  box-shadow:0 12px 28px rgba(21,128,61,.2) !important;
  transition:transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
}
.stButton>button:hover {
  transform:translateY(-1px);
  filter:saturate(1.08);
  box-shadow:0 16px 34px rgba(21,128,61,.28) !important;
}

.result-card {
  border-radius:8px;
  padding:1.25rem;
  background:#FFFFFF;
  border:1px solid var(--line);
  box-shadow:0 18px 50px rgba(15,23,42,.08);
  animation:scaleIn .35s ease both;
}
.result-top { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:.9rem; }
.badge {
  display:inline-flex;
  align-items:center;
  border-radius:999px;
  padding:.35rem .7rem;
  font-size:.78rem;
  font-weight:800;
}
.badge.healthy { background:#DCFCE7; color:#166534; border:1px solid #BBF7D0; }
.badge.disease { background:#FEF3C7; color:#92400E; border:1px solid #FDE68A; }
.badge.unknown { background:#FEE2E2; color:#991B1B; border:1px solid #FECACA; }
.label-kicker { color:var(--muted); font-size:.78rem; font-weight:800; text-transform:uppercase; }
.result-title { margin:.25rem 0 .25rem; color:var(--forest); font-size:clamp(1.45rem, 3vw, 2.2rem); line-height:1.12; font-weight:800; }
.result-subtitle { color:var(--muted); line-height:1.6; margin:0; }
.confidence-row { display:flex; align-items:end; gap:.45rem; margin:1rem 0 .45rem; }
.confidence-value { color:var(--forest); font-size:2.3rem; line-height:1; font-weight:800; }
.confidence-label { color:var(--muted); font-size:.9rem; padding-bottom:.2rem; }
.progress-shell {
  width:100%;
  height:12px;
  border-radius:999px;
  background:#E2E8F0;
  overflow:hidden;
}
.progress-fill {
  height:100%;
  border-radius:999px;
  background:linear-gradient(90deg, var(--leaf), var(--lime));
  animation:fillBar .9s ease both;
}
.detail-grid {
  display:grid;
  grid-template-columns:repeat(2, minmax(0,1fr));
  gap:.7rem;
  margin-top:1rem;
}
.detail {
  border:1px solid #E2E8F0;
  border-radius:8px;
  padding:.75rem;
  background:#F8FAFC;
}
.detail span { display:block; color:var(--muted); font-size:.72rem; font-weight:800; text-transform:uppercase; }
.detail strong { display:block; color:var(--ink); margin-top:.2rem; overflow-wrap:anywhere; }
.recommend {
  border:1px solid var(--line);
  border-left:4px solid var(--leaf);
  border-radius:8px;
  background:#FFFFFF;
  padding:1rem 1.15rem;
  margin-top:1rem;
}
.recommend h4 { color:var(--forest); margin:0 0 .55rem; font-size:1rem; }
.recommend ul { margin:.45rem 0 0 1.1rem; padding:0; color:#334155; line-height:1.75; }
.disclaimer {
  margin-top:.75rem;
  color:var(--muted);
  font-size:.86rem;
  line-height:1.55;
}
.pred-row {
  display:grid;
  grid-template-columns:1.35rem minmax(0,1fr) 4.5rem;
  gap:.65rem;
  align-items:center;
  padding:.55rem 0;
  border-bottom:1px solid #E2E8F0;
}
.pred-row:last-child { border-bottom:0; }
.pred-rank { color:#166534; font-weight:800; }
.pred-name { color:#334155; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pred-pct { color:var(--forest); font-weight:800; text-align:right; }
.mini-bar { grid-column:2 / 4; height:7px; border-radius:999px; background:#E2E8F0; overflow:hidden; }
.mini-fill { height:100%; border-radius:999px; background:linear-gradient(90deg, var(--leaf), var(--lime)); }

.warning-card {
  border:1px solid #FDE68A;
  border-left:4px solid var(--amber);
  background:#FFFBEB;
  color:#78350F;
  border-radius:8px;
  padding:.95rem 1rem;
  line-height:1.55;
}
.error-card {
  border:1px solid #FECACA;
  border-left:4px solid var(--red);
  background:#FEF2F2;
  color:#7F1D1D;
  border-radius:8px;
  padding:.95rem 1rem;
  line-height:1.55;
}
.empty-state {
  border:1px solid var(--line);
  border-radius:8px;
  padding:2rem;
  background:rgba(255,255,255,.72);
  text-align:center;
  color:var(--muted);
}
.insight-grid {
  display:grid;
  grid-template-columns:repeat(4, minmax(0,1fr));
  gap:.8rem;
  margin:1rem 0;
}
.insight-stat {
  border:1px solid var(--line);
  border-radius:8px;
  background:#FFFFFF;
  padding:1rem;
}
.insight-stat strong { display:block; color:var(--forest); font-size:1.45rem; }
.insight-stat span { color:var(--muted); font-size:.78rem; font-weight:700; }
.footer {
  margin-top:2rem;
  padding:1.3rem 0 0;
  border-top:1px solid var(--line);
  color:var(--muted);
  font-size:.82rem;
  text-align:center;
}

@media (max-width: 760px) {
  .block-container { padding:1rem .8rem 2.5rem !important; }
  .hero { padding:1.35rem; min-height:0; }
  .hero-metrics, .detail-grid, .insight-grid { grid-template-columns:1fr; }
  .section-head { display:block; }
  .result-top { align-items:flex-start; flex-direction:column; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CLASSES
# ─────────────────────────────────────────────
CLASSES = sorted([
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot',
    'Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy',
    'Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch',
    'Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight',
    'Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
])

DB = {
    'Apple_scab':('medium','Fungal lesions causing dark scabby spots on leaves and fruit.','Apply fungicide. Remove infected leaves. Prune for better airflow.'),
    'Black_rot':('high','Fungal disease with circular brown spots and dark margins.','Prune infected branches. Apply copper fungicide. Remove mummified fruit.'),
    'Cedar_apple_rust':('medium','Orange rust spots on upper leaf surface, tubes below.','Apply myclobutanil in spring. Remove nearby cedar trees if possible.'),
    'Powdery_mildew':('medium','White powdery fungal coating on the leaf surface.','Apply sulfur or neem-based fungicide. Improve air circulation.'),
    'Cercospora_leaf_spot':('medium','Circular gray spots with dark purple borders.','Apply fungicide. Remove crop debris. Avoid overhead watering.'),
    'Common_rust':('medium','Orange-brown rust pustules on both leaf surfaces.','Apply fungicide early in season. Use resistant corn varieties.'),
    'Northern_Leaf_Blight':('high','Long tan cigar-shaped lesions with dark wavy borders.','Apply fungicide. Rotate crops annually. Use resistant varieties.'),
    'Esca':('high','Tiger-stripe yellowing pattern with dark wood streaks.','No cure available. Remove infected vines to prevent spread.'),
    'Leaf_blight':('high','Brown water-soaked lesions spreading rapidly across leaves.','Apply copper fungicide. Avoid wet foliage. Improve drainage.'),
    'Haunglongbing':('high','Yellow shoots and blotchy mottled asymmetric leaves.','No cure. Remove infected trees immediately to stop spread.'),
    'Bacterial_spot':('medium','Small water-soaked spots turning angular and brown.','Apply copper spray. Avoid working with plants when foliage is wet.'),
    'Early_blight':('medium','Dark concentric-ring spots — like a target on leaves.','Apply fungicide. Remove lower infected leaves. Mulch around base.'),
    'Late_blight':('high','Water-soaked lesions turning dark brown, white mold below.','Apply fungicide immediately. Remove and destroy infected plants.'),
    'Leaf_Mold':('medium','Pale yellow spots above, olive-green mold on underside.','Improve ventilation. Reduce humidity. Apply appropriate fungicide.'),
    'Septoria_leaf_spot':('medium','Small circular spots — dark border, light gray center.','Apply fungicide. Remove infected lower leaves promptly.'),
    'Spider_mites':('medium','Yellow stippling with fine silky webbing on leaves.','Apply miticide or neem oil. Increase ambient humidity around plants.'),
    'Target_Spot':('medium','Circular brown spots with distinct concentric rings.','Apply fungicide. Improve air circulation around the plant canopy.'),
    'Yellow_Leaf_Curl_Virus':('high','Upward leaf curl, yellowing margins, stunted growth.','No cure. Remove infected plants. Control whitefly vectors.'),
    'mosaic_virus':('high','Mosaic light-dark green mottled pattern, distorted leaves.','No cure. Remove infected plants. Control aphid vectors immediately.'),
    'Leaf_scorch':('medium','Brown scorched margins and leaf tips due to stress.','Improve irrigation consistency. Mulch around base of plant.'),
    'healthy':('none','No disease detected. The plant appears healthy.','Continue regular watering, fertilization, and routine monitoring.'),
}

def get_info(cls):
    plant, cond = cls.split('___')
    if cond == 'healthy':
        return plant, 'Healthy', DB['healthy']
    for key, val in DB.items():
        if key.lower().replace('_','') in cond.lower().replace('_','').replace(' ',''):
            return plant, cond.replace('_',' '), val
    return plant, cond.replace('_',' '), ('medium', f'Disease detected in {plant}.', 'Consult an agricultural expert.')


# ─────────────────────────────────────────────
# GOAD TRANSFORMS — same as notebook
# ─────────────────────────────────────────────
_norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
GOAD_TRANSFORMS = [
    transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomRotation((90,90)), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomRotation((180,180)), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomRotation((270,270)), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomRotation((90,90)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), _norm]),
]
M = len(GOAD_TRANSFORMS)


# ─────────────────────────────────────────────
# MODEL — exact match to notebook
# ─────────────────────────────────────────────
class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.network = models.efficientnet_b3(weights=None)
        in_features  = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)

    def extract_features(self, xb):
        features = self.network.features(xb)
        features = self.network.avgpool(features)
        features = features.flatten(1)
        logits   = self.network.classifier(features)
        return features, logits


# ─────────────────────────────────────────────
# LOAD MODEL + ANOMALY PARAMS
# ─────────────────────────────────────────────
@st.cache_resource
def load_everything():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model      = EfficientNetB3(38)
    model_path = os.path.join(BASE_DIR, 'Classification-based Anomaly Detection', 'plant-disease-model.pth')
    model_ok   = False
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            model_ok = True
            log.info('Model loaded successfully from %s', model_path)
        except Exception as e:
            log.error('Failed to load model from %s: %s', model_path, e)

    # Anomaly params
    anomaly_params = None
    anom_path      = os.path.join(BASE_DIR, 'Classification-based Anomaly Detection', 'anomaly_params.pkl')
    if os.path.exists(anom_path):
        try:
            with open(anom_path, 'rb') as f:
                anomaly_params = pickle.load(f)
            log.info('Anomaly params loaded from %s', anom_path)
        except Exception as e:
            log.error('Failed to load anomaly params from %s: %s', anom_path, e)

    return model, device, model_ok, anomaly_params


# ─────────────────────────────────────────────
# INFERENCE TRANSFORM
# ─────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


# ─────────────────────────────────────────────
# GOAD SCORE — Mahalanobis distance
# ─────────────────────────────────────────────
def compute_goad_score(pil_image, model, device, cluster_centers, cov_inv, epsilon=0.1):
    log_probs = []
    for m_idx, t in enumerate(GOAD_TRANSFORMS):
        tensor = t(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat, _ = model.extract_features(tensor)
        feat = feat[0].cpu().numpy()
        mahal_dists = np.array([
            float((feat - cluster_centers[m_prime]) @ cov_inv @ (feat - cluster_centers[m_prime]))
            for m_prime in range(M)
        ])
        neg_d  = -mahal_dists
        exp_d  = np.exp(neg_d - neg_d.max())
        numer  = exp_d[m_idx] + epsilon
        denom  = exp_d.sum() + M * epsilon
        log_probs.append(np.log(numer / denom))
    return -sum(log_probs)


# ─────────────────────────────────────────────
# FULL PIPELINE — anomaly check + classification
# ─────────────────────────────────────────────
def run_pipeline(pil_image, model, device, anomaly_params):
    tensor = TRANSFORM(pil_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        feats, logits = model.extract_features(tensor)
        probs         = torch.softmax(logits, dim=1)[0]
        msp_conf      = probs.max().item()

    result = {
        'is_anomaly': False, 'reason': '',
        'msp_conf': msp_conf, 'goad_score': 0.0,
        'top5': None, 'plant': None, 'condition': None,
        'anomaly_checked': anomaly_params is not None
    }

    if anomaly_params:
        # MSP check
        if msp_conf < anomaly_params['msp_threshold']:
            result.update({
                'is_anomaly': True,
                'reason': f'MSP: confidence too low ({msp_conf*100:.1f}% < {anomaly_params["msp_threshold"]*100:.1f}%)',
            })
            return result

        # GOAD check
        g_score = compute_goad_score(
            pil_image, model, device,
            anomaly_params['cluster_centers'],
            anomaly_params['cov_inv'],
            anomaly_params.get('epsilon', 0.1)
        )
        result['goad_score'] = g_score

        if g_score > anomaly_params['goad_threshold']:
            result.update({
                'is_anomaly': True,
                'reason': f'GOAD: unusual pattern detected (score {g_score:.2f} > threshold {anomaly_params["goad_threshold"]:.2f})',
            })
            return result

    # Normal classification
    top5p, top5i = torch.topk(probs, 5)
    top5 = [{'class': CLASSES[i.item()], 'prob': p.item()}
            for p, i in zip(top5p.cpu(), top5i.cpu())]
    plant, condition = top5[0]['class'].split('___')
    result.update({'top5': top5, 'plant': plant, 'condition': condition})
    return result


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for k, v in [('stage',0),('result',None),('image',None),
             ('fname',''),('fsize',0),('imgwh',(0,0))]:
    if k not in st.session_state:
        st.session_state[k] = v

model, device, model_ok, anomaly_params = load_everything()
stage = st.session_state.stage


def pretty_class(raw_class):
    plant, condition = raw_class.split('___')
    return plant.replace('_', ' '), condition.replace('_', ' ')


def reset_app():
    for k, v in [('stage',0),('result',None),('image',None),
                 ('fname',''),('fsize',0),('imgwh',(0,0))]:
        st.session_state[k] = v


def result_status(confidence, is_healthy, is_anomaly=False):
    if is_anomaly or confidence < 55:
        return "Unknown / low confidence", "unknown"
    if is_healthy:
        return "Healthy", "healthy"
    return "Disease detected", "disease"


def recommendation_items(is_healthy, is_low_confidence):
    items = [
        "Check the leaf under natural light before making decisions.",
        "Capture another image from a different angle for comparison.",
        "Consult an agricultural expert for confirmation.",
    ]
    if is_low_confidence:
        items.insert(1, "Use a sharper close-up image with only one leaf in frame.")
    if not is_healthy:
        items.append("Isolate affected plants if disease symptoms are visible.")
    return items


def insight_image(filename, caption):
    path = os.path.join(BASE_DIR, 'Classification-based Anomaly Detection', filename)
    st.markdown(f'<div class="card"><h3 style="margin:0 0 .8rem;color:#0B2E1A;font-size:1rem">{caption}</h3>', unsafe_allow_html=True)
    if os.path.exists(path):
        st.image(path, width="stretch")
    else:
        st.info(f"{filename} not found")
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────
anomaly_label = "MSP + GOAD active" if anomaly_params else "Anomaly guard unavailable"
st.markdown(f"""
<section class="hero">
  <div class="hero-content">
    <div class="eyebrow">CNN • ResNet • EfficientNet-B3 • PlantVillage</div>
    <h1>PlantGuard AI</h1>
    <h2>AI-powered plant disease detection</h2>
    <p>Upload a plant leaf image and let the model detect possible diseases with confidence insights.</p>
  </div>
  <div class="hero-metrics">
    <div class="metric"><strong>38</strong><span>PlantVillage classes</span></div>
    <div class="metric"><strong>70K+</strong><span>training images</span></div>
    <div class="metric"><strong>EffNet-B3</strong><span>model backbone</span></div>
    <div class="metric"><strong>{anomaly_label}</strong><span>quality check</span></div>
  </div>
</section>
""", unsafe_allow_html=True)

st.markdown('<div class="tabs-wrap">', unsafe_allow_html=True)
tab_diag, tab_insights = st.tabs(["Diagnosis", "Model insights"])
st.markdown('</div>', unsafe_allow_html=True)


with tab_diag:
    st.markdown("""
    <div class="section-head">
      <div>
        <h3>Upload and analyze</h3>
        <p>A clear close-up leaf photo works best. Supported formats: JPG, PNG, JPEG.</p>
      </div>
      <span class="model-pill">EfficientNet-B3 • 300 x 300 input</span>
    </div>
    """, unsafe_allow_html=True)

    upload_col, status_col = st.columns([1.08, .92], gap="large")

    with upload_col:
        st.markdown("""
        <div class="upload-card">
          <div class="upload-icon">↑</div>
          <h3 class="upload-title">Upload a leaf image</h3>
          <p class="upload-copy">Drag and drop a file, or browse from your device.</p>
          <div class="format-row">
            <span class="chip">JPG</span>
            <span class="chip">PNG</span>
            <span class="chip">JPEG</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload a leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

    with status_col:
        st.markdown(f"""
        <div class="status-card">
          <span class="model-pill">Model status</span>
          <div class="detail-grid">
            <div class="detail"><span>Prediction model</span><strong>{'Ready' if model_ok else 'Model file missing'}</strong></div>
            <div class="detail"><span>Device</span><strong>{str(device).upper()}</strong></div>
            <div class="detail"><span>Anomaly check</span><strong>{'Ready' if anomaly_params else 'Not loaded'}</strong></div>
            <div class="detail"><span>Classes</span><strong>{len(CLASSES)}</strong></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if not model_ok:
            st.markdown("""
            <div class="error-card">
              The model file could not be loaded. Expected:
              <strong>src/Classification-based Anomaly Detection/plant-disease-model.pth</strong>
            </div>
            """, unsafe_allow_html=True)
        elif not anomaly_params:
            st.markdown("""
            <div class="warning-card">
              Anomaly parameters were not found. Predictions can still run, but low-quality or unrelated images may be less reliably rejected.
            </div>
            """, unsafe_allow_html=True)

    if uploaded:
        try:
            img = Image.open(uploaded).convert('RGB')
            if uploaded.name != st.session_state.fname:
                st.session_state.update({
                    'image': img,
                    'fname': uploaded.name,
                    'fsize': round(uploaded.size / 1024, 1),
                    'imgwh': img.size,
                    'stage': 1,
                    'result': None
                })
        except Exception as exc:
            st.markdown(f"""
            <div class="error-card">
              We could not read this image. Please upload a valid JPG, PNG, or JPEG file.
              <br><small>{escape(str(exc))}</small>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.image:
        img = st.session_state.image
        width, height = st.session_state.imgwh
        st.markdown("""
        <div class="section-head">
          <div>
            <h3>Leaf preview</h3>
            <p>Review the image, then run the model when you are ready.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        preview_col, action_col = st.columns([1, 1], gap="large")
        with preview_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(img, caption="Uploaded leaf image", width="stretch")
            st.markdown(f"""
            <div class="detail-grid">
              <div class="detail"><span>Filename</span><strong>{escape(st.session_state.fname)}</strong></div>
              <div class="detail"><span>File size</span><strong>{st.session_state.fsize} KB</strong></div>
              <div class="detail"><span>Dimensions</span><strong>{width} x {height} px</strong></div>
              <div class="detail"><span>Color mode</span><strong>RGB</strong></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with action_col:
            st.markdown("""
            <div class="card">
              <h3 style="margin:0 0 .4rem;color:#0B2E1A">Ready for diagnosis</h3>
              <p style="margin:0 0 1rem;color:#64748B;line-height:1.6">
                PlantGuard AI will resize the image, normalize it with ImageNet statistics,
                run anomaly checks when available, and classify it with EfficientNet-B3.
              </p>
            </div>
            """, unsafe_allow_html=True)

            analyze = st.button("Analyze leaf", width="stretch", disabled=not model_ok)
            if analyze:
                with st.spinner("Analyzing leaf image with PlantGuard AI..."):
                    progress = st.progress(0)
                    progress.progress(25)
                    time.sleep(0.1)
                    progress.progress(58)
                    st.session_state.result = run_pipeline(img, model, device, anomaly_params)
                    progress.progress(100)
                    st.session_state.stage = 3 if st.session_state.result['is_anomaly'] else 4
                    time.sleep(0.1)

            if st.session_state.result is not None:
                if st.button("Analyze another image", width="stretch"):
                    reset_app()
                    st.rerun()

        if st.session_state.result:
            res = st.session_state.result
            st.markdown("""
            <div class="section-head">
              <div>
                <h3>Prediction result</h3>
                <p>AI-assisted diagnosis with confidence and next-step guidance.</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

            image_col, result_col = st.columns([.9, 1.1], gap="large")
            with image_col:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.image(img, caption="Image used for prediction", width="stretch")
                st.markdown("""
                <p class="preview-note">For best results, use one leaf in frame, natural lighting, and minimal background clutter.</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with result_col:
                if res['is_anomaly']:
                    confidence = res['msp_conf'] * 100
                    status_text, status_class = result_status(confidence, False, True)
                    st.markdown(f"""
                    <div class="result-card">
                      <div class="result-top">
                        <div>
                          <div class="label-kicker">Prediction label</div>
                          <h3 class="result-title">Unknown image pattern</h3>
                        </div>
                        <span class="badge {status_class}">{status_text}</span>
                      </div>
                      <p class="result-subtitle">{escape(res.get('reason', 'The image appears outside the known training distribution.'))}</p>
                      <div class="confidence-row">
                        <div class="confidence-value">{confidence:.1f}%</div>
                        <div class="confidence-label">model confidence</div>
                      </div>
                      <div class="progress-shell"><div class="progress-fill" style="width:{min(confidence, 100):.1f}%"></div></div>
                      <div class="detail-grid">
                        <div class="detail"><span>Predicted class</span><strong>Unknown / anomaly</strong></div>
                        <div class="detail"><span>Model used</span><strong>EfficientNet-B3 + MSP/GOAD</strong></div>
                        <div class="detail"><span>MSP score</span><strong>{res['msp_conf']*100:.2f}%</strong></div>
                        <div class="detail"><span>GOAD score</span><strong>{res['goad_score']:.4f}</strong></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    items = recommendation_items(False, True)
                else:
                    top = res['top5'][0]
                    plant, disease, (severity, description, treatment) = get_info(top['class'])
                    confidence = top['prob'] * 100
                    is_healthy = severity == 'none' or 'healthy' in top['class'].lower()
                    low_conf = confidence < 70
                    status_text, status_class = result_status(confidence, is_healthy)
                    friendly_title = "Healthy leaf" if is_healthy else disease
                    explanation = (
                        "The model did not detect a disease pattern in this image."
                        if is_healthy else
                        f"The model found visual patterns associated with {disease.lower()}."
                    )
                    st.markdown(f"""
                    <div class="result-card">
                      <div class="result-top">
                        <div>
                          <div class="label-kicker">{escape(plant.replace('_', ' '))}</div>
                          <h3 class="result-title">{escape(friendly_title)}</h3>
                        </div>
                        <span class="badge {status_class}">{status_text}</span>
                      </div>
                      <p class="result-subtitle">{escape(explanation)} {escape(description)}</p>
                      <div class="confidence-row">
                        <div class="confidence-value">{confidence:.1f}%</div>
                        <div class="confidence-label">confidence</div>
                      </div>
                      <div class="progress-shell"><div class="progress-fill" style="width:{min(confidence, 100):.1f}%"></div></div>
                      <div class="detail-grid">
                        <div class="detail"><span>Predicted class</span><strong>{escape(top['class'])}</strong></div>
                        <div class="detail"><span>Model used</span><strong>EfficientNet-B3</strong></div>
                        <div class="detail"><span>Confidence</span><strong>{confidence:.2f}%</strong></div>
                        <div class="detail"><span>Recommendation</span><strong>{escape(treatment)}</strong></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if low_conf:
                        st.markdown("""
                        <div class="warning-card">
                          Confidence is below the recommended review threshold. Try a clearer image and verify with an expert.
                        </div>
                        """, unsafe_allow_html=True)
                    items = recommendation_items(is_healthy, low_conf)

                st.markdown('<div class="recommend"><h4>Suggested Next Steps</h4><ul>', unsafe_allow_html=True)
                for item in items:
                    st.markdown(f"<li>{escape(item)}</li>", unsafe_allow_html=True)
                st.markdown("""
                </ul>
                <p class="disclaimer">This prediction is AI-assisted and should be verified by an expert.</p>
                </div>
                """, unsafe_allow_html=True)

                if not res['is_anomaly'] and res['top5']:
                    st.markdown('<div class="card" style="margin-top:1rem"><h3 style="margin:0 0 .65rem;color:#0B2E1A;font-size:1rem">Top predictions</h3>', unsafe_allow_html=True)
                    for idx, pred in enumerate(res['top5'], start=1):
                        p_plant, p_condition = pretty_class(pred['class'])
                        pct = pred['prob'] * 100
                        st.markdown(f"""
                        <div class="pred-row">
                          <div class="pred-rank">{idx}</div>
                          <div class="pred-name">{escape(p_plant)} • {escape(p_condition)}</div>
                          <div class="pred-pct">{pct:.1f}%</div>
                          <div class="mini-bar"><div class="mini-fill" style="width:{min(pct, 100):.1f}%"></div></div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
          <strong>Upload a plant leaf image to begin.</strong><br>
          PlantGuard AI supports common crops including apple, grape, tomato, potato, corn, peach, cherry, pepper, strawberry, blueberry, orange, soybean, squash, and raspberry.
        </div>
        """, unsafe_allow_html=True)


with tab_insights:
    st.markdown("""
    <div class="section-head">
      <div>
        <h3>Model insights</h3>
        <p>Training and anomaly-detection context for students, researchers, and reviewers.</p>
      </div>
    </div>
    <div class="insight-grid">
      <div class="insight-stat"><strong>99%+</strong><span>validation accuracy</span></div>
      <div class="insight-stat"><strong>38</strong><span>classification labels</span></div>
      <div class="insight-stat"><strong>3</strong><span>training epochs</span></div>
      <div class="insight-stat"><strong>MSP+GOAD</strong><span>anomaly method</span></div>
    </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns(2, gap="large")
    with c_left:
        insight_image('confusion_matrix.png', 'Confusion matrix')
    with c_right:
        insight_image('per_class_f1.png', 'Per-class F1 score')

    c_ov, c_dist = st.columns(2, gap="large")
    with c_ov:
        insight_image('overfitting_analysis .png', 'Overfitting analysis')
    with c_dist:
        insight_image('class_distribution.png', 'Class distribution')

    c_msp, c_goad = st.columns(2, gap="large")
    with c_msp:
        insight_image('msp_distribution.png', 'MSP score distribution')
        if anomaly_params:
            st.markdown(f'<div class="detail"><span>MSP threshold</span><strong>{anomaly_params["msp_threshold"]*100:.2f}%</strong></div>', unsafe_allow_html=True)
    with c_goad:
        insight_image('goad_distribution .png', 'GOAD score distribution')
        if anomaly_params:
            st.markdown(f'<div class="detail"><span>GOAD threshold</span><strong>{anomaly_params["goad_threshold"]:.4f}</strong></div>', unsafe_allow_html=True)


st.markdown("""
<div class="footer">
  PlantGuard AI uses EfficientNet-B3 on PlantVillage classes with optional MSP + GOAD anomaly detection. Results are decision support, not agricultural certainty.
</div>
""", unsafe_allow_html=True)
