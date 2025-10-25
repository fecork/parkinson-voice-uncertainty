import json

# Cargar notebook
with open('research/cnn_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Buscar celda con optimizador
for i, cell in enumerate(nb['cells']):
    source = cell.get('source', [])
    source_str = ''.join(source)
    if 'optimizer_final = optim.Adam' in source_str:
        print(f"Encontrada en celda {i}")
        print("Contenido:")
        for j, line in enumerate(source):
            print(f"  {j}: {repr(line)}")
        break

