# ============================================================
# USAR CHECKPOINT EN NOTEBOOK
# ============================================================

print("=" * 70)
print("USANDO CHECKPOINT EN NOTEBOOK")
print("=" * 70)

# Ejecutar el script de creación de checkpoint
exec(open("research/create_initial_checkpoint.py").read())

print("\n🚀 Ahora puedes usar el checkpoint en tu notebook:")
print("""
# En tu notebook, reemplaza la llamada a optimize_cnn2d con:

optuna_results = optimize_cnn2d(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    input_shape=(1, 65, 41),
    n_trials=30,
    n_epochs_per_trial=20,
    device=device,
    save_dir=str(optuna_results_dir),
    checkpoint_dir="checkpoints",  # ← NUEVO
    resume=True  # ← NUEVO
)
""")

print("\n💡 Ventajas del checkpointing:")
print("   - ✅ Continúa desde donde se quedó")
print("   - ✅ Guarda progreso automáticamente")
print("   - ✅ No pierde trabajo si se corta la GPU")
print("   - ✅ Puede reanudar múltiples veces")
print("   - ✅ Guarda mejores parámetros en tiempo real")

print("\n" + "=" * 70)
print("CHECKPOINT LISTO PARA USAR")
print("=" * 70)
