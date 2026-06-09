"""
Comprime manuscript_ijacsa_blind.pdf reduciendo imagenes a 150 DPI
manteniendo texto vectorial intacto (buscable, copiable).
"""
import fitz, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src  = os.path.join(ROOT, 'ijacsa_paper', 'manuscript_ijacsa_blind.pdf')
dst  = os.path.join(ROOT, 'ijacsa_paper', 'manuscript_ijacsa_blind_compressed.pdf')

doc = fitz.open(src)
orig_mb = os.path.getsize(src) / 1024 / 1024
print(f"Original: {orig_mb:.1f} MB  ({len(doc)} paginas)")

# Comprimir imagenes embebidas en el PDF a JPEG calidad 72
# El texto vectorial NO se toca — sigue siendo buscable
for i, page in enumerate(doc):
    img_list = page.get_images(full=True)
    for img_info in img_list:
        xref = img_info[0]
        try:
            # Extraer imagen, comprimir a JPEG 72, reinsertar
            img_dict = doc.extract_image(xref)
            if img_dict and img_dict.get('colorspace', 0) in (1, 3):
                from PIL import Image
                import io
                pil_img = Image.open(io.BytesIO(img_dict['image']))
                # Reducir resolucion si es muy grande
                w, h = pil_img.size
                max_px = 1800  # ~150 DPI a tamano carta
                if max(w, h) > max_px:
                    scale = max_px / max(w, h)
                    pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                buf = io.BytesIO()
                pil_img = pil_img.convert('RGB')
                pil_img.save(buf, format='JPEG', quality=72, optimize=True)
                buf.seek(0)
                doc.update_image(xref, stream=buf.read())
        except Exception as e:
            pass  # saltar imagenes que no se pueden comprimir

# Guardar con compresion maxima
doc.save(dst,
         garbage=4,       # eliminar objetos no referenciados
         deflate=True,    # comprimir streams
         clean=True,      # limpiar contenido
         deflate_images=True,
         deflate_fonts=True)
doc.close()

new_mb = os.path.getsize(dst) / 1024 / 1024
print(f"Comprimido: {new_mb:.1f} MB")
print(f"Reduccion: {(1 - new_mb/orig_mb)*100:.0f}%")
print(f"Guardado en: {dst}")
