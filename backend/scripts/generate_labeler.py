# backend/scripts/generate_labeler.py
import os
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "dataset" / "images"
META_CSV = BASE / "dataset" / "metadata.csv"
OUT_HTML = BASE / "dataset" / "labeler.html"
FEATURES_NPY = BASE / "models" / "features.npy"
OUT_CSV = BASE / "dataset" / "clusters.csv"

parser = argparse.ArgumentParser(description="Generate clusters + HTML labeler")
parser.add_argument("--n_clusters", type=int, default=30, help="Number of clusters to create")
args = parser.parse_args()

# --- Load metadata ---
if not META_CSV.exists():
    raise SystemExit(f"metadata.csv not found at {META_CSV}")

df = pd.read_csv(META_CSV)

items = []
for idx, row in df.iterrows():
    img_rel = str(row["image_path"])
    abs_path = str((IMG_DIR / Path(img_rel).name).resolve())
    rel_path = f"images/{Path(img_rel).name}"

    items.append({
        "item_id": int(row.get("item_id", idx + 1)),
        "name": str(row.get("name", "")),
        "category": str(row.get("category", "")),
        "occasion": str(row.get("occasion", "")),
        "body_type": str(row.get("body_type", "")),
        "image_path": abs_path,   # for CSV export
        "rel_path": rel_path      # for HTML preview
    })

# --- ResNet50 features ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def resnet_feature(abs_path):
    try:
        img = Image.open(abs_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet(x).cpu().numpy().flatten()
        return feat
    except Exception as e:
        print(f"⚠️ Feature extraction failed for {abs_path}: {e}")
        return np.zeros(2048)

# --- Load or compute features ---
if FEATURES_NPY.exists():
    feats = np.load(FEATURES_NPY)
    feats_used = feats[:len(items)]
else:
    feats_used = np.vstack([resnet_feature(it["image_path"]) for it in items])
    FEATURES_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_NPY, feats_used)

# --- Clustering ---
km = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
labels = km.fit_predict(feats_used)

for i, it in enumerate(items):
    it["cluster"] = int(labels[i])

# --- Save CSV (correct schema, absolute paths) ---
df_out = pd.DataFrame(items)[[
    "cluster", "item_id", "name", "category", "occasion", "body_type", "image_path"
]]
df_out.to_csv(OUT_CSV, index=False)
print(f"✅ Clustered metadata saved to {OUT_CSV}")

# --- Generate HTML labeler with cluster-level apply buttons ---
data_json = json.dumps(items)
categories = ["Dress", "Shirt","Top","Pants", "Traditional", "Jacket", "T-shirt", "Skirt", "Shorts", "Unknown"]
occasions = ["Casual", "Party", "Formal", "Sporty", "Everyday", "Unknown"]
body_types = ["Athletic", "Hourglass", "Pear", "Inverted Triangle", "Rectangle", "Unknown"]

html = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>Metadata Labeler</title>
<style>
body {font-family: Arial; margin: 20px;}
.cluster {border:1px solid #ddd; margin:12px; padding:8px; border-radius:8px;}
.cluster h3 {margin: 4px 0;}
.controls {margin-bottom:8px;}
.image-item {display:inline-block; width:150px; margin:6px; text-align:center; vertical-align:top;}
.image-item img {width:120px; height:120px; object-fit:cover; border:1px solid #ccc; display:block; margin:0 auto;}
.button-row {margin-top:6px;}
.top-actions {margin-bottom:18px;}
</style>
</head><body>
<h2>Labeler - cluster defaults + apply to cluster</h2>

<div class="top-actions">
  <button id="applyAllOverwrite">Apply all cluster defaults to all clusters (overwrite)</button>
  <button id="applyAllUnset">Apply all cluster defaults to all clusters (only unset)</button>
  <span style="margin-left:12px;color:#555">Tip: choose cluster defaults then use the buttons to apply.</span>
</div>

<div id="container"></div>
<button id="exportBtn">Export CSV</button>
<script>
const items = __DATA__;
const categories = __CATEGORIES__;
const occasions = __OCCASIONS__;
const body_types = __BODYTYPES__;

// group by cluster
const groups = {};
items.forEach(it => {
  if(!groups[it.cluster]) groups[it.cluster] = [];
  groups[it.cluster].push(it);
});

const container = document.getElementById('container');

// build UI per cluster
Object.keys(groups).sort((a,b)=>a-b).forEach(cid => {
  const cluster = groups[cid];
  const div = document.createElement('div');
  div.className = 'cluster';
  div.id = 'cluster_' + cid;

  // header + cluster defaults + apply buttons
  let headerHtml = '<h3>Cluster ' + cid + ' (' + cluster.length + ' items)</h3>';
  headerHtml += '<div class="controls"><b>Cluster defaults:</b> ';
  headerHtml += 'Category: <select id="cluster_cat_'+cid+'">'+ categories.map(c=>'<option>'+c+'</option>').join('') +'</select> ';
  headerHtml += 'Occasion: <select id="cluster_occ_'+cid+'">'+ occasions.map(c=>'<option>'+c+'</option>').join('') +'</select> ';
  headerHtml += 'Body: <select id="cluster_body_'+cid+'">'+ body_types.map(c=>'<option>'+c+'</option>').join('') +'</select> ';
  headerHtml += '<button onclick="applyClusterDefaults('+cid+', false)">Apply defaults (only unset)</button> ';
  headerHtml += '<button onclick="applyClusterDefaults('+cid+', true)">Apply defaults (overwrite)</button>';
  headerHtml += '</div>';
  div.innerHTML = headerHtml;

  // images
  const imagesDiv = document.createElement('div');
  cluster.forEach(it => {
    const imgBlock = document.createElement('div');
    imgBlock.className = 'image-item';
    imgBlock.setAttribute('data-relpath', it.rel_path);
    imgBlock.innerHTML = ''
      + '<img src="'+it.rel_path+'" alt="'+it.item_id+'">'
      + '<div style="font-size:12px; margin-top:6px;">'+it.name+'</div>'
      + '<div class="button-row">'
      + '<select data-prop="category"><option value="">(use cluster default)</option>'
          + categories.map(c=>'<option '+(c==it.category?'selected':'')+'>'+c+'</option>').join('')
        + '</select>'
      + '<br/>'
      + '<select data-prop="occasion"><option value="">(use cluster default)</option>'
          + occasions.map(c=>'<option '+(c==it.occasion?'selected':'')+'>'+c+'</option>').join('')
        + '</select>'
      + '<br/>'
      + '<select data-prop="body_type"><option value="">(use cluster default)</option>'
          + body_types.map(c=>'<option '+(c==it.body_type?'selected':'')+'>'+c+'</option>').join('')
        + '</select>'
      + '</div>';
    imagesDiv.appendChild(imgBlock);
  });

  div.appendChild(imagesDiv);
  container.appendChild(div);
});

// Function to apply cluster defaults to images in cluster
function applyClusterDefaults(cid, overwrite=false){
  const clusterDiv = document.getElementById('cluster_' + cid);
  if(!clusterDiv) return;
  const defaultCat = clusterDiv.querySelector('#cluster_cat_' + cid).value;
  const defaultOcc = clusterDiv.querySelector('#cluster_occ_' + cid).value;
  const defaultBody = clusterDiv.querySelector('#cluster_body_' + cid).value;
  const imgItems = clusterDiv.querySelectorAll('.image-item');
  imgItems.forEach(imgDiv => {
    const catSel = imgDiv.querySelector('select[data-prop="category"]');
    const occSel = imgDiv.querySelector('select[data-prop="occasion"]');
    const bodySel = imgDiv.querySelector('select[data-prop="body_type"]');
    if(overwrite || !catSel.value) catSel.value = defaultCat;
    if(overwrite || !occSel.value) occSel.value = defaultOcc;
    if(overwrite || !bodySel.value) bodySel.value = defaultBody;
  });
}

// global apply actions
document.getElementById('applyAllOverwrite').onclick = () => {
  Object.keys(groups).forEach(cid => applyClusterDefaults(cid, true));
};
document.getElementById('applyAllUnset').onclick = () => {
  Object.keys(groups).forEach(cid => applyClusterDefaults(cid, false));
};

// Export CSV: use individual selects if non-empty, else cluster default, else original
document.getElementById('exportBtn').onclick = () => {
  let csv = "cluster,item_id,name,category,occasion,body_type,image_path\\n";

  Object.keys(groups).forEach(cid => {
    const clusterDiv = document.getElementById('cluster_' + cid);
    const clusterDefaults = {
      category: clusterDiv.querySelector('#cluster_cat_' + cid).value,
      occasion: clusterDiv.querySelector('#cluster_occ_' + cid).value,
      body_type: clusterDiv.querySelector('#cluster_body_' + cid).value
    };

    const imgItems = clusterDiv.querySelectorAll('.image-item');
    imgItems.forEach(imgDiv => {
      const rel = imgDiv.getAttribute('data-relpath');
      // find corresponding item in items array
      const it = items.find(x => x.rel_path === rel);
      if(!it) return;
      const catSel = imgDiv.querySelector('select[data-prop="category"]');
      const occSel = imgDiv.querySelector('select[data-prop="occasion"]');
      const bodySel = imgDiv.querySelector('select[data-prop="body_type"]');
      const finalCat = (catSel && catSel.value) || clusterDefaults.category || it.category;
      const finalOcc = (occSel && occSel.value) || clusterDefaults.occasion || it.occasion;
      const finalBody = (bodySel && bodySel.value) || clusterDefaults.body_type || it.body_type;

      csv += [it.cluster, it.item_id, '"' + it.name.replace(/"/g,'""') + '"', finalCat, finalOcc, finalBody, it.image_path].join(',') + "\\n";
    });
  });

  const blob = new Blob([csv], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'metadata_corrected.csv';
  a.click();
};
</script></body></html>
"""

html = html.replace("__DATA__", data_json)
html = html.replace("__CATEGORIES__", json.dumps(categories))
html = html.replace("__OCCASIONS__", json.dumps(occasions))
html = html.replace("__BODYTYPES__", json.dumps(body_types))

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Labeler generated at: {OUT_HTML}")
print("👉 Open with: cd backend/dataset && python -m http.server 8000")
print("   Then visit http://localhost:8000/labeler.html in browser")
