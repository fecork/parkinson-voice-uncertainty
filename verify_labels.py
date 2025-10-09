#!/usr/bin/env python3
"""
Script de Verificación de Etiquetas
====================================
Verifica que las etiquetas estén correctas antes de entrenar.
"""

from pathlib import Path
import torch
from modules.dataset import build_full_pipeline

# Configuración
HC_DIR = "data/vowels_healthy"
PD_DIR = "data/vowels_pk"

print("\n" + "=" * 70)
print("VERIFICACIÓN DE ETIQUETAS")
print("=" * 70)

# Cargar archivos
hc_files = list(Path(HC_DIR).glob("*.egg"))
pd_files = list(Path(PD_DIR).glob("*.egg"))

print(f"\n📁 Archivos:")
print(f"  - HC: {len(hc_files)} archivos")
print(f"  - PD: {len(pd_files)} archivos")

# Procesar
print("\n🔄 Procesando HC...")
hc_result = build_full_pipeline(hc_files, max_files=None)
hc_dataset = hc_result["torch_ds"]

print("\n🔄 Procesando PD...")
pd_result = build_full_pipeline(pd_files, max_files=None)
pd_dataset = pd_result["torch_ds"]

# VERIFICAR ETIQUETAS ANTES DE CORREGIR
print("\n" + "=" * 70)
print("❌ ETIQUETAS INCORRECTAS (sin corregir)")
print("=" * 70)

hc_labels_before = hc_dataset.y_task.numpy()
pd_labels_before = pd_dataset.y_task.numpy()

print(f"\n🟢 HC Dataset:")
print(f"   Total segmentos: {len(hc_labels_before)}")
print(f"   Etiquetas únicas: {set(hc_labels_before.tolist())}")
print(f"   Etiqueta 0 (HC): {(hc_labels_before == 0).sum()} segmentos")
print(f"   Etiqueta 1 (PD): {(hc_labels_before == 1).sum()} segmentos ⚠️ INCORRECTO!")

print(f"\n🔴 PD Dataset:")
print(f"   Total segmentos: {len(pd_labels_before)}")
print(f"   Etiquetas únicas: {set(pd_labels_before.tolist())}")
print(f"   Etiqueta 0 (HC): {(pd_labels_before == 0).sum()} segmentos ⚠️ INCORRECTO!")
print(f"   Etiqueta 1 (PD): {(pd_labels_before == 1).sum()} segmentos")

# CORREGIR ETIQUETAS
print("\n" + "=" * 70)
print("✅ CORRIGIENDO ETIQUETAS")
print("=" * 70)

for i in range(len(hc_dataset.y_task)):
    hc_dataset.y_task[i] = 0

for i in range(len(pd_dataset.y_task)):
    pd_dataset.y_task[i] = 1

# VERIFICAR DESPUÉS DE CORREGIR
print("\n✅ ETIQUETAS CORRECTAS (después de corregir)")
print("=" * 70)

hc_labels_after = hc_dataset.y_task.numpy()
pd_labels_after = pd_dataset.y_task.numpy()

print(f"\n🟢 HC Dataset:")
print(f"   Total segmentos: {len(hc_labels_after)}")
print(f"   Etiquetas únicas: {set(hc_labels_after.tolist())}")
print(f"   Etiqueta 0 (HC): {(hc_labels_after == 0).sum()} segmentos ✓")
print(f"   Etiqueta 1 (PD): {(hc_labels_after == 1).sum()} segmentos")

print(f"\n🔴 PD Dataset:")
print(f"   Total segmentos: {len(pd_labels_after)}")
print(f"   Etiquetas únicas: {set(pd_labels_after.tolist())}")
print(f"   Etiqueta 0 (HC): {(pd_labels_after == 0).sum()} segmentos")
print(f"   Etiqueta 1 (PD): {(pd_labels_after == 1).sum()} segmentos ✓")

# Combinar
from torch.utils.data import ConcatDataset

combined = ConcatDataset([hc_dataset, pd_dataset])

print("\n" + "=" * 70)
print("📊 DATASET COMBINADO")
print("=" * 70)

# Contar etiquetas en dataset combinado
all_labels = []
for i in range(len(combined)):
    item = combined[i]
    all_labels.append(item["y_task"].item())

all_labels = torch.tensor(all_labels)
print(f"\nTotal segmentos: {len(all_labels)}")
print(f"Etiqueta 0 (HC): {(all_labels == 0).sum().item()} segmentos")
print(f"Etiqueta 1 (PD): {(all_labels == 1).sum().item()} segmentos")
print(f"Balance: HC={len(hc_dataset)}, PD={len(pd_dataset)}")
print(f"Ratio: PD/HC = {len(pd_dataset) / len(hc_dataset):.2f}")

# Verificar que todos HC sean 0 y todos PD sean 1
hc_correct = all((all_labels[i] == 0) for i in range(len(hc_dataset)))
pd_correct = all((all_labels[len(hc_dataset) + i] == 1) for i in range(len(pd_dataset)))

print("\n" + "=" * 70)
if hc_correct and pd_correct:
    print("✅ VERIFICACIÓN EXITOSA: Todas las etiquetas son correctas")
else:
    print("❌ ERROR: Hay etiquetas incorrectas")
print("=" * 70 + "\n")

# Mostrar ejemplos
print("\n📋 EJEMPLOS DE ETIQUETAMIENTO:")
print("-" * 70)
print(f"{'Archivo':<30} {'Speaker':<10} {'Pitch':<10} {'Label'}")
print("-" * 70)

# 3 ejemplos HC
for i in range(min(3, len(hc_dataset))):
    item = hc_dataset[i]
    meta = item["meta"]
    label = item["y_task"].item()
    print(
        f"{meta.filename:<30} {meta.subject_id:<10} {meta.condition:<10} {label} (HC)"
    )

print()

# 3 ejemplos PD
for i in range(min(3, len(pd_dataset))):
    item = pd_dataset[i]
    meta = item["meta"]
    label = item["y_task"].item()
    print(
        f"{meta.filename:<30} {meta.subject_id:<10} {meta.condition:<10} {label} (PD)"
    )

print("-" * 70)
