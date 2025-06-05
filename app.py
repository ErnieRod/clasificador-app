import streamlit as st 
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s
from PIL import Image

# Ruta al modelo
MODEL_PATH = "./best_efficientnetv2_s.pth"
CLASSES = ["Categoria_1", "Categoria_2", "Categoria_3", "Categoria_4", "Categoria_5"] 

# Comentarios asociados a cada clase
COMMENTS = {
    "Categoria_1": "✅ Puede seguir operando el componente, no presenta ninguna falla estructural.",
    "Categoria_2": "ℹ️ Puede seguir operando el componente, sin embargo no descuide el análisis de aceites.",
    "Categoria_3": "⚠️ Contacte a personal de Confiabilidad, el componente presenta indicios de falla.",
    "Categoria_4": "⚠️ El componente debe removerse antes de que llegue a la falla catastrófica.",
    "Categoria_5": "🛑 El componente tiene falla catastrófica. Remover con URGENCIA."
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = efficientnet_v2_s(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

model = load_model()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Título personalizado
st.title("Clasificador de Fallas en Componentes Mecánicos: Mando Finales by ERZ")

uploaded_file = st.file_uploader("Carga una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Imagen cargada", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = probs.argmax()
    pred_score = probs[pred_idx]
    pred_class = CLASSES[pred_idx]

    if pred_score < 0.85:
        st.error("⚠️ La imagen cargada no puede ser clasificada. No corresponde a un tapón magnético o presenta características inusuales.")
    else:
        st.success(f"🔎 **Categoría Predicha:** {pred_class}")
        st.info(COMMENTS[pred_class])

