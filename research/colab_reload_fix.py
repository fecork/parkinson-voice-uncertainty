# ============================================================
# COLAB RELOAD FIX - Limpiar caché y recargar módulos
# ============================================================

print("=" * 70)
print("COLAB RELOAD FIX - Limpiando caché y recargando módulos")
print("=" * 70)

# Asegurar instalación de optuna
import subprocess
import sys

print("🔧 Asegurando instalación de optuna...")
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "optuna>=3.0.0"], check=True
    )
    print("✅ Optuna instalado correctamente")
except Exception as e:
    print(f"⚠️  Error instalando optuna: {e}")

# Limpiar sys.modules y recargar
import importlib

print("\n🧹 Limpiando caché de módulos...")
mods_to_reload = [
    "modules.core.cnn2d_optuna_wrapper",
    "modules.core.optuna_optimization",
    "modules.models.cnn2d.model",
    "modules.core.dataset",
    "modules.core.utils",
]

for mod in mods_to_reload:
    if mod in sys.modules:
        del sys.modules[mod]
        print(f"   ✅ Limpiado: {mod}")

print("\n🔄 Recargando módulos...")
try:
    import modules.core.cnn2d_optuna_wrapper as cnn2d_wrap

    importlib.reload(cnn2d_wrap)
    print("   ✅ CNN2D Optuna Wrapper recargado")

    import modules.core.optuna_optimization as opt_opt

    importlib.reload(opt_opt)
    print("   ✅ Optuna Optimization recargado")

    import modules.models.cnn2d.model as cnn2d_model

    importlib.reload(cnn2d_model)
    print("   ✅ CNN2D Model recargado")

    import modules.core.dataset as dataset

    importlib.reload(dataset)
    print("   ✅ Dataset recargado")

    import modules.core.utils as utils

    importlib.reload(utils)
    print("   ✅ Utils recargado")

except Exception as e:
    print(f"   ❌ Error recargando módulos: {e}")

print("\n🔍 Verificando imports...")
try:
    from modules.core.cnn2d_optuna_wrapper import optimize_cnn2d, CNN2DOptunaWrapper
    from optuna.exceptions import TrialPruned

    print("   ✅ Imports funcionando correctamente")
    print(f"   ✅ TrialPruned disponible: {TrialPruned}")
except Exception as e:
    print(f"   ❌ Error en imports: {e}")

print("\n" + "=" * 70)
print("RECARGA COMPLETADA - Ahora puedes ejecutar el notebook")
print("=" * 70)
