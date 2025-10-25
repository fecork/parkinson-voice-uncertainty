# ============================================================
# VERIFICACIÓN DEL FIX - Confirmar que no hay referencias problemáticas
# ============================================================

print("=" * 70)
print("VERIFICACIÓN DEL FIX - Buscando referencias problemáticas")
print("=" * 70)

import os
import subprocess


# Función para ejecutar grep
def run_grep(pattern, path="."):
    try:
        result = subprocess.run(
            ["grep", "-r", "-n", pattern, path],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()
    except:
        return "Error ejecutando grep"


print("🔍 Buscando referencias problemáticas...")

# 1. Buscar optuna.TrialPruned
print("\n1. Buscando 'optuna.TrialPruned':")
optuna_trialpruned = run_grep("optuna\.TrialPruned")
if optuna_trialpruned:
    print("   ❌ ENCONTRADAS referencias problemáticas:")
    print(f"   {optuna_trialpruned}")
else:
    print("   ✅ NO se encontraron referencias a 'optuna.TrialPruned'")

# 2. Buscar except optuna.TrialPruned
print("\n2. Buscando 'except optuna.TrialPruned':")
except_optuna_trialpruned = run_grep("except.*optuna\.TrialPruned")
if except_optuna_trialpruned:
    print("   ❌ ENCONTRADAS referencias problemáticas:")
    print(f"   {except_optuna_trialpruned}")
else:
    print("   ✅ NO se encontraron referencias a 'except optuna.TrialPruned'")

# 3. Buscar imports correctos
print("\n3. Verificando imports correctos:")
from_optuna_exceptions = run_grep("from optuna.exceptions import TrialPruned")
if from_optuna_exceptions:
    print("   ✅ ENCONTRADOS imports correctos:")
    print(f"   {from_optuna_exceptions}")
else:
    print(
        "   ⚠️  NO se encontraron imports de 'from optuna.exceptions import TrialPruned'"
    )

# 4. Buscar uso directo de TrialPruned
print("\n4. Verificando uso directo de 'TrialPruned':")
trialpruned_direct = run_grep("raise TrialPruned")
if trialpruned_direct:
    print("   ✅ ENCONTRADO uso directo correcto:")
    print(f"   {trialpruned_direct}")
else:
    print("   ⚠️  NO se encontró uso directo de 'raise TrialPruned'")

print("\n" + "=" * 70)
print("VERIFICACIÓN COMPLETADA")
print("=" * 70)

# Resumen
if not optuna_trialpruned and not except_optuna_trialpruned:
    print("🎉 FIX APLICADO CORRECTAMENTE - No hay referencias problemáticas")
else:
    print("⚠️  AÚN HAY REFERENCIAS PROBLEMÁTICAS - Revisar archivos mencionados")
