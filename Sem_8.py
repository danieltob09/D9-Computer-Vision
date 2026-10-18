from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops


# =========================================================
# 1. CONFIGURACIÓN
# =========================================================

IMAGE_PATH = Path(r"A:\PROYECTOS\D9\San_F.jpg")
IMAGE_SIZE = (224, 224)

# Reducción de ruido
H = 10
H_COLOR = 10
TEMPLATE_WINDOW = 7
SEARCH_WINDOW = 21

# CLAHE: mejora local del contraste
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# GLCM: análisis de texturas
GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [
    0,
    np.pi / 4,
    np.pi / 2,
    3 * np.pi / 4
]
GLCM_LEVELS = 32

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# 2. VALIDAR RUTA Y BUSCAR IMÁGENES
# =========================================================

print("========== VALIDACIÓN DE RUTA ==========")
print(f"Ruta seleccionada: {IMAGE_PATH}")
print(f"¿Existe el archivo?: {IMAGE_PATH.is_file()}")
print(f"¿Existe la carpeta?: {IMAGE_PATH.parent.is_dir()}")

if not IMAGE_PATH.is_file():
    search_folder = Path(r"A:\PROYECTOS\D9")

    print(f"\nBuscando imágenes en: {search_folder}")

    if search_folder.is_dir():
        found_images = [
            file
            for file in search_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        ]

        if found_images:
            for file in found_images:
                print(f"  - {file}")
        else:
            print("No se encontraron imágenes.")
    else:
        print("La carpeta de búsqueda no existe.")

    raise FileNotFoundError(
        f"No se encontró la imagen: {IMAGE_PATH}\n"
        "Actualiza IMAGE_PATH con una de las rutas encontradas."
    )


# =========================================================
# 3. CARGAR IMAGEN
# =========================================================

img_bgr = cv2.imread(str(IMAGE_PATH))

if img_bgr is None:
    raise ValueError(
        f"El archivo existe, pero OpenCV no pudo leerlo: {IMAGE_PATH}"
    )

original_height, original_width = img_bgr.shape[:2]

print(f"\nDimensiones originales: "
      f"{original_width} x {original_height}")

# Redimensionar para mantener un tamaño uniforme
img_bgr = cv2.resize(
    img_bgr,
    IMAGE_SIZE,
    interpolation=cv2.INTER_AREA
)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# =========================================================
# 4. REDUCCIÓN DE RUIDO
# =========================================================

denoised = cv2.fastNlMeansDenoisingColored(
    img_bgr,
    None,
    H,
    H_COLOR,
    TEMPLATE_WINDOW,
    SEARCH_WINDOW
)


# =========================================================
# 5. MEJORA DEL CONTRASTE CON CLAHE
# =========================================================

lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)

l_channel, a_channel, b_channel = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP_LIMIT,
    tileGridSize=CLAHE_TILE_GRID_SIZE
)

l_enhanced = clahe.apply(l_channel)

lab_enhanced = cv2.merge(
    (l_enhanced, a_channel, b_channel)
)

enhanced_bgr = cv2.cvtColor(
    lab_enhanced,
    cv2.COLOR_LAB2BGR
)

enhanced_rgb = cv2.cvtColor(
    enhanced_bgr,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# 6. COMPARACIÓN VISUAL
# =========================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

images = [
    (img_rgb, "Imagen original"),
    (
        cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB),
        "Reducción de ruido"
    ),
    (enhanced_rgb, "Ruido reducido + CLAHE")
]

for ax, (image_data, title) in zip(axes, images):
    ax.imshow(image_data)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()


# =========================================================
# 7. CLASIFICACIÓN CON MOBILENETV2
# =========================================================

print("\nCargando MobileNetV2...")

model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=True
)


def classify_image(image_rgb, model, top=5):
    """Clasifica una imagen RGB con MobileNetV2."""

    resized = cv2.resize(
        image_rgb,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    batch = np.expand_dims(
        resized.astype(np.float32),
        axis=0
    )

    processed = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(batch.copy())
    )

    predictions = model.predict(
        processed,
        verbose=0
    )

    return (
        tf.keras.applications.mobilenet_v2
        .decode_predictions(predictions, top=top)[0]
    )


print("\n========== CLASIFICACIÓN ==========")

for name, image_data in [
    ("ORIGINAL", img_rgb),
    ("PROCESADA", enhanced_rgb)
]:
    print(f"\n{name}")

    for _, label, probability in classify_image(
        image_data, model, top=5
    ):
        print(f"  {label}: {probability:.2%}")


# =========================================================
# 8. EXTRACCIÓN DE TEXTURAS MEDIANTE GLCM
# =========================================================

def extract_glcm_features(image_rgb):
    """Calcula estadísticas de textura usando una matriz GLCM."""

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    # Cuantizar 256 niveles de gris a 32 niveles
    quantized = (
        gray // (256 // GLCM_LEVELS)
    ).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True
    )

    properties = [
        "contrast",
        "correlation",
        "homogeneity",
        "energy"
    ]

    features = {}

    for prop in properties:
        values = graycoprops(glcm, prop)
        features[prop] = float(np.mean(values))

    return features


original_features = extract_glcm_features(img_rgb)
processed_features = extract_glcm_features(enhanced_rgb)

print("\n========== CARACTERÍSTICAS GLCM ==========")

for feature in original_features:
    original_value = original_features[feature]
    processed_value = processed_features[feature]

    difference = processed_value - original_value

    print(f"\n{feature.capitalize()}")
    print(f"  Original:    {original_value:.5f}")
    print(f"  Procesada:   {processed_value:.5f}")
    print(f"  Diferencia:  {difference:+.5f}")


# =========================================================
# 9. GUARDAR IMAGEN PROCESADA
# =========================================================

output_path = IMAGE_PATH.with_name(
    f"{IMAGE_PATH.stem}_procesada.jpg"
)

success = cv2.imwrite(
    str(output_path),
    enhanced_bgr
)

if success:
    print(f"\nImagen procesada guardada en:\n{output_path}")
else:
    raise OSError(
        f"No se pudo guardar la imagen en: {output_path}"
    )


# =========================================================
# 10. RESUMEN FINAL
# =========================================================

print("\n========== PROCESO FINALIZADO ==========")
print(f"Entrada: {IMAGE_PATH}")
print(f"Salida:  {output_path}")
print("Procesamiento de imagen completado.")
print("Clasificación y análisis de texturas completados.")

print(
    "\nNota: MobileNetV2 utiliza clases generales de ImageNet; "
    "no es un detector especializado de grietas o baches."
)
