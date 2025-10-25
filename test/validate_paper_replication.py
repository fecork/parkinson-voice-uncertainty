"""
Script de Validación de Replicación del Paper Ibarra 2023
==========================================================

Valida si el notebook cnn_training.ipynb cumple con los requisitos
del paper de Ibarra et al. (2023) para el primer experimento (vocal /a/).

Uso:
    python test/validate_paper_replication.py research/cnn_training.ipynb
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import nbformat


class PaperValidator:
    """Validador de cumplimiento del paper."""

    def __init__(self, requirements_path: str = "test/paper_requirements.json"):
        """
        Inicializar validador.

        Args:
            requirements_path: Ruta al archivo de requisitos del paper
        """
        with open(requirements_path, "r") as f:
            self.requirements = json.load(f)

        self.results = {
            "preprocessing": [],
            "architecture": [],
            "training": [],
            "validation": [],
            "hyperparameters": [],
        }

        self.notebook_content = ""

    def load_notebook(self, notebook_path: str):
        """Cargar y parsear notebook."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Concatenar todo el código
        code_cells = [
            cell["source"] for cell in nb.cells if cell["cell_type"] == "code"
        ]
        self.notebook_content = "\n".join(code_cells)

    def validate_preprocessing(self) -> List[Dict]:
        """Validar parámetros de preprocesamiento."""
        results = []
        prep_req = self.requirements["preprocessing"]

        # Sampling rate
        if "44100" in self.notebook_content or "44.1" in self.notebook_content:
            results.append(
                {
                    "check": "Sampling rate",
                    "expected": "44.1 kHz",
                    "status": "PASS",
                    "message": "Sampling rate correcto",
                }
            )
        else:
            results.append(
                {
                    "check": "Sampling rate",
                    "expected": "44.1 kHz",
                    "status": "FAIL",
                    "message": "No se encuentra sampling rate de 44.1 kHz",
                }
            )

        # Mel filters
        if "n_mels=65" in self.notebook_content or "n_mel=65" in self.notebook_content:
            results.append(
                {
                    "check": "Mel filters",
                    "expected": "65",
                    "status": "PASS",
                    "message": "Número de filtros Mel correcto",
                }
            )
        else:
            results.append(
                {
                    "check": "Mel filters",
                    "expected": "65",
                    "status": "WARNING",
                    "message": "No se encuentra n_mels=65 explícitamente",
                }
            )

        # Output shape
        if "65" in self.notebook_content and "41" in self.notebook_content:
            results.append(
                {
                    "check": "Output shape",
                    "expected": "(65, 41)",
                    "status": "PASS",
                    "message": "Dimensiones de salida presentes",
                }
            )
        else:
            results.append(
                {
                    "check": "Output shape",
                    "expected": "(65, 41)",
                    "status": "FAIL",
                    "message": "No se encuentran dimensiones (65, 41)",
                }
            )

        # Vocal filtering
        vocal_filter = prep_req.get("vocal_filter", "a")
        if (
            f"vowel.*{vocal_filter}" in self.notebook_content.lower()
            or f"vocal.*{vocal_filter}" in self.notebook_content.lower()
        ):
            results.append(
                {
                    "check": "Vocal filtering",
                    "expected": f"Solo vocal /{vocal_filter}/",
                    "status": "PASS",
                    "message": f"Filtrado de vocal /{vocal_filter}/ presente",
                }
            )
        else:
            results.append(
                {
                    "check": "Vocal filtering",
                    "expected": f"Solo vocal /{vocal_filter}/",
                    "status": "FAIL",
                    "message": f"No se encuentra filtrado de vocal /{vocal_filter}/",
                }
            )

        return results

    def validate_architecture(self) -> List[Dict]:
        """Validar arquitectura CNN2D."""
        results = []

        # CNN2D model
        if "CNN2D" in self.notebook_content or "class CNN2D" in self.notebook_content:
            results.append(
                {
                    "check": "Model class",
                    "expected": "CNN2D",
                    "status": "PASS",
                    "message": "Modelo CNN2D encontrado",
                }
            )
        else:
            results.append(
                {
                    "check": "Model class",
                    "expected": "CNN2D",
                    "status": "FAIL",
                    "message": "No se encuentra clase CNN2D",
                }
            )

        # Convolutional blocks
        conv_pattern = r"Conv2d.*\(.*32.*\)|Conv2d.*\(.*64.*\)"
        if re.search(conv_pattern, self.notebook_content):
            results.append(
                {
                    "check": "Conv blocks",
                    "expected": "2 bloques Conv2D (32, 64 filtros)",
                    "status": "PASS",
                    "message": "Bloques convolucionales presentes",
                }
            )
        else:
            results.append(
                {
                    "check": "Conv blocks",
                    "expected": "2 bloques Conv2D (32, 64 filtros)",
                    "status": "WARNING",
                    "message": "No se detectan bloques convolucionales explícitamente",
                }
            )

        # Dropout
        if "Dropout" in self.notebook_content:
            dropout_values = re.findall(
                r"Dropout\s*\(\s*p?=?\s*(0\.\d+)", self.notebook_content
            )
            if dropout_values:
                results.append(
                    {
                        "check": "Dropout",
                        "expected": "0.3 (conv), 0.5 (fc)",
                        "status": "PASS",
                        "message": f"Dropout encontrado: {set(dropout_values)}",
                    }
                )
            else:
                results.append(
                    {
                        "check": "Dropout",
                        "expected": "0.3 (conv), 0.5 (fc)",
                        "status": "WARNING",
                        "message": "Dropout presente pero valores no detectados",
                    }
                )
        else:
            results.append(
                {
                    "check": "Dropout",
                    "expected": "0.3 (conv), 0.5 (fc)",
                    "status": "FAIL",
                    "message": "No se encuentra Dropout",
                }
            )

        # Input shape
        if "(65, 41)" in self.notebook_content or "65.*41" in self.notebook_content:
            results.append(
                {
                    "check": "Input shape",
                    "expected": "(1, 65, 41)",
                    "status": "PASS",
                    "message": "Dimensiones de entrada correctas",
                }
            )
        else:
            results.append(
                {
                    "check": "Input shape",
                    "expected": "(1, 65, 41)",
                    "status": "FAIL",
                    "message": "Dimensiones de entrada no encontradas",
                }
            )

        return results

    def validate_training(self) -> List[Dict]:
        """Validar configuración de entrenamiento."""
        results = []
        train_req = self.requirements["training"]

        # Optimizer
        if (
            "optim.SGD" in self.notebook_content
            or "torch.optim.SGD" in self.notebook_content
        ):
            results.append(
                {
                    "check": "Optimizer",
                    "expected": "SGD",
                    "status": "PASS",
                    "message": "Optimizer SGD encontrado",
                }
            )
        elif (
            "optim.Adam" in self.notebook_content
            or "torch.optim.Adam" in self.notebook_content
        ):
            results.append(
                {
                    "check": "Optimizer",
                    "expected": "SGD",
                    "status": "FAIL",
                    "message": "Encontrado Adam en lugar de SGD",
                }
            )
        else:
            results.append(
                {
                    "check": "Optimizer",
                    "expected": "SGD",
                    "status": "FAIL",
                    "message": "No se encuentra optimizer",
                }
            )

        # Learning rate
        lr_pattern = r"lr\s*=\s*0\.1\b"
        if re.search(lr_pattern, self.notebook_content):
            results.append(
                {
                    "check": "Learning rate",
                    "expected": "0.1",
                    "status": "PASS",
                    "message": "Learning rate inicial correcto (0.1)",
                }
            )
        else:
            results.append(
                {
                    "check": "Learning rate",
                    "expected": "0.1",
                    "status": "FAIL",
                    "message": "Learning rate no es 0.1",
                }
            )

        # Momentum
        momentum_pattern = r"momentum\s*=\s*0\.9"
        if re.search(momentum_pattern, self.notebook_content):
            results.append(
                {
                    "check": "Momentum",
                    "expected": "0.9",
                    "status": "PASS",
                    "message": "Momentum correcto (0.9)",
                }
            )
        else:
            results.append(
                {
                    "check": "Momentum",
                    "expected": "0.9",
                    "status": "FAIL",
                    "message": "Momentum no encontrado o incorrecto",
                }
            )

        # Scheduler
        if "StepLR" in self.notebook_content:
            results.append(
                {
                    "check": "Scheduler",
                    "expected": "StepLR",
                    "status": "PASS",
                    "message": "Scheduler StepLR encontrado",
                }
            )
        elif "ReduceLROnPlateau" in self.notebook_content:
            results.append(
                {
                    "check": "Scheduler",
                    "expected": "StepLR",
                    "status": "FAIL",
                    "message": "Encontrado ReduceLROnPlateau en lugar de StepLR",
                }
            )
        else:
            results.append(
                {
                    "check": "Scheduler",
                    "expected": "StepLR",
                    "status": "FAIL",
                    "message": "No se encuentra scheduler",
                }
            )

        # Step size
        step_pattern = r"step_size\s*=\s*10"
        if re.search(step_pattern, self.notebook_content):
            results.append(
                {
                    "check": "Scheduler step_size",
                    "expected": "10",
                    "status": "PASS",
                    "message": "Step size correcto (10)",
                }
            )
        else:
            results.append(
                {
                    "check": "Scheduler step_size",
                    "expected": "10",
                    "status": "FAIL",
                    "message": "Step size no encontrado o incorrecto",
                }
            )

        # Gamma
        gamma_pattern = r"gamma\s*=\s*0\.1"
        if re.search(gamma_pattern, self.notebook_content):
            results.append(
                {
                    "check": "Scheduler gamma",
                    "expected": "0.1",
                    "status": "PASS",
                    "message": "Gamma correcto (0.1)",
                }
            )
        else:
            results.append(
                {
                    "check": "Scheduler gamma",
                    "expected": "0.1",
                    "status": "FAIL",
                    "message": "Gamma no encontrado o incorrecto",
                }
            )

        # Loss function
        if "CrossEntropyLoss" in self.notebook_content:
            if (
                "weight" in self.notebook_content
                and "class_weight" in self.notebook_content.lower()
            ):
                results.append(
                    {
                        "check": "Loss function",
                        "expected": "Weighted CrossEntropyLoss",
                        "status": "PASS",
                        "message": "CrossEntropyLoss con pesos encontrado",
                    }
                )
            else:
                results.append(
                    {
                        "check": "Loss function",
                        "expected": "Weighted CrossEntropyLoss",
                        "status": "FAIL",
                        "message": "CrossEntropyLoss sin pesos balanceados",
                    }
                )
        else:
            results.append(
                {
                    "check": "Loss function",
                    "expected": "Weighted CrossEntropyLoss",
                    "status": "FAIL",
                    "message": "CrossEntropyLoss no encontrado",
                }
            )

        # Metric
        if (
            "f1" in self.notebook_content.lower()
            and "macro" in self.notebook_content.lower()
        ):
            results.append(
                {
                    "check": "Metric",
                    "expected": "F1-macro",
                    "status": "PASS",
                    "message": "Métrica F1-macro encontrada",
                }
            )
        else:
            results.append(
                {
                    "check": "Metric",
                    "expected": "F1-macro",
                    "status": "WARNING",
                    "message": "F1-macro no detectado explícitamente",
                }
            )

        return results

    def validate_validation_strategy(self) -> List[Dict]:
        """Validar estrategia de validación."""
        results = []
        val_req = self.requirements["validation"]

        # K-Fold
        if (
            "KFold" in self.notebook_content
            or "StratifiedKFold" in self.notebook_content
        ):
            if (
                "n_splits=10" in self.notebook_content
                or "n_splits = 10" in self.notebook_content
            ):
                results.append(
                    {
                        "check": "Cross-validation",
                        "expected": "10-fold CV",
                        "status": "PASS",
                        "message": "10-fold cross-validation encontrado",
                    }
                )
            else:
                results.append(
                    {
                        "check": "Cross-validation",
                        "expected": "10-fold CV",
                        "status": "WARNING",
                        "message": "K-Fold encontrado pero n_splits != 10",
                    }
                )
        else:
            if "train_test_split" in self.notebook_content:
                results.append(
                    {
                        "check": "Cross-validation",
                        "expected": "10-fold CV",
                        "status": "FAIL",
                        "message": "Usa train_test_split en lugar de K-Fold",
                    }
                )
            else:
                results.append(
                    {
                        "check": "Cross-validation",
                        "expected": "10-fold CV",
                        "status": "FAIL",
                        "message": "No se encuentra estrategia de validación",
                    }
                )

        # Speaker-based split
        if (
            "speaker" in self.notebook_content.lower()
            or "hablante" in self.notebook_content.lower()
        ):
            results.append(
                {
                    "check": "Speaker stratification",
                    "expected": "Split por hablante",
                    "status": "PASS",
                    "message": "Estratificación por hablante detectada",
                }
            )
        else:
            results.append(
                {
                    "check": "Speaker stratification",
                    "expected": "Split por hablante",
                    "status": "FAIL",
                    "message": "No se encuentra estratificación por hablante",
                }
            )

        # Averaging metrics
        if "mean" in self.notebook_content.lower() and (
            "fold" in self.notebook_content.lower()
            or "cv" in self.notebook_content.lower()
        ):
            results.append(
                {
                    "check": "Metric averaging",
                    "expected": "Promedio de métricas de folds",
                    "status": "PASS",
                    "message": "Promedio de métricas detectado",
                }
            )
        else:
            results.append(
                {
                    "check": "Metric averaging",
                    "expected": "Promedio de métricas de folds",
                    "status": "WARNING",
                    "message": "Promedio de métricas no detectado",
                }
            )

        return results

    def validate_hyperparameters(self) -> List[Dict]:
        """Validar espacio de búsqueda de hiperparámetros."""
        results = []
        hp_req = self.requirements["hyperparameters"]

        # Batch size
        batch_sizes = hp_req["batch_size"]
        if all(str(bs) in self.notebook_content for bs in batch_sizes):
            results.append(
                {
                    "check": "Batch size search space",
                    "expected": str(batch_sizes),
                    "status": "PASS",
                    "message": f"Batch sizes correctos: {batch_sizes}",
                }
            )
        else:
            results.append(
                {
                    "check": "Batch size search space",
                    "expected": str(batch_sizes),
                    "status": "WARNING",
                    "message": "No todos los batch sizes encontrados",
                }
            )

        # Dropout
        dropout_values = hp_req["dropout_conv"]
        if any(str(d) in self.notebook_content for d in dropout_values):
            results.append(
                {
                    "check": "Dropout search space",
                    "expected": str(dropout_values),
                    "status": "PASS",
                    "message": f"Valores de dropout presentes",
                }
            )
        else:
            results.append(
                {
                    "check": "Dropout search space",
                    "expected": str(dropout_values),
                    "status": "WARNING",
                    "message": "Valores de dropout no detectados",
                }
            )

        # FC units
        fc_units = hp_req["fc_units"]
        if any(str(u) in self.notebook_content for u in fc_units):
            results.append(
                {
                    "check": "FC units search space",
                    "expected": str(fc_units),
                    "status": "PASS",
                    "message": f"FC units presentes",
                }
            )
        else:
            results.append(
                {
                    "check": "FC units search space",
                    "expected": str(fc_units),
                    "status": "WARNING",
                    "message": "FC units no detectados",
                }
            )

        return results

    def run_validation(self, notebook_path: str) -> Dict:
        """Ejecutar validación completa."""
        print("=" * 70)
        print("VALIDACION DE REPLICACION DEL PAPER IBARRA 2023")
        print("=" * 70)
        print(f"\nNotebook: {notebook_path}")
        print(f"Experimento: {self.requirements['experiment']}\n")

        # Cargar notebook
        self.load_notebook(notebook_path)

        # Ejecutar validaciones
        self.results["preprocessing"] = self.validate_preprocessing()
        self.results["architecture"] = self.validate_architecture()
        self.results["training"] = self.validate_training()
        self.results["validation"] = self.validate_validation_strategy()
        self.results["hyperparameters"] = self.validate_hyperparameters()

        # Mostrar resultados
        self.print_results()

        # Generar reporte
        self.generate_report(notebook_path)

        return self.results

    def print_results(self):
        """Imprimir resultados en consola."""
        categories = [
            ("PREPROCESAMIENTO", self.results["preprocessing"]),
            ("ARQUITECTURA", self.results["architecture"]),
            ("ENTRENAMIENTO", self.results["training"]),
            ("VALIDACION", self.results["validation"]),
            ("HIPERPARAMETROS", self.results["hyperparameters"]),
        ]

        total_checks = 0
        passed_checks = 0

        for category_name, checks in categories:
            print(f"\n[{category_name}]")
            for check in checks:
                status_icon = (
                    "[PASS]"
                    if check["status"] == "PASS"
                    else "[WARN]"
                    if check["status"] == "WARNING"
                    else "[FAIL]"
                )
                print(f"{status_icon} {check['check']}: {check['message']}")

                total_checks += 1
                if check["status"] == "PASS":
                    passed_checks += 1

        # Resumen
        compliance_rate = (
            (passed_checks / total_checks * 100) if total_checks > 0 else 0
        )
        print("\n" + "=" * 70)
        print(
            f"RESUMEN: {passed_checks}/{total_checks} criterios cumplidos ({compliance_rate:.0f}%)"
        )

        if compliance_rate >= 80:
            print("ESTADO: CUMPLE CON EL PAPER")
        elif compliance_rate >= 50:
            print("ESTADO: CUMPLIMIENTO PARCIAL")
        else:
            print("ESTADO: NO CUMPLE CON EL PAPER")

        print("=" * 70)

    def generate_report(self, notebook_path: str):
        """Generar reporte en Markdown."""
        report_path = Path("test/PAPER_VALIDATION_REPORT.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Reporte de Validación del Paper Ibarra 2023\n\n")
            f.write(f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Notebook**: {notebook_path}\n")
            f.write(f"**Experimento**: {self.requirements['experiment']}\n\n")

            f.write("## Resultados por Categoría\n\n")

            categories = [
                ("Preprocesamiento", self.results["preprocessing"]),
                ("Arquitectura", self.results["architecture"]),
                ("Entrenamiento", self.results["training"]),
                ("Validación", self.results["validation"]),
                ("Hiperparámetros", self.results["hyperparameters"]),
            ]

            for category_name, checks in categories:
                f.write(f"### {category_name}\n\n")
                f.write("| Check | Esperado | Estado | Mensaje |\n")
                f.write("|-------|----------|--------|----------|\n")

                for check in checks:
                    status_icon = (
                        "✅"
                        if check["status"] == "PASS"
                        else "⚠️"
                        if check["status"] == "WARNING"
                        else "❌"
                    )
                    f.write(
                        f"| {check['check']} | {check['expected']} | {status_icon} {check['status']} | {check['message']} |\n"
                    )

                f.write("\n")

            # Recomendaciones
            f.write("## Recomendaciones de Corrección\n\n")

            all_fails = []
            for category_checks in self.results.values():
                all_fails.extend([c for c in category_checks if c["status"] == "FAIL"])

            if all_fails:
                for i, check in enumerate(all_fails, 1):
                    f.write(f"{i}. **{check['check']}**: {check['message']}\n")
            else:
                f.write("No se encontraron problemas críticos.\n")

        print(f"\nReporte guardado en: {report_path}")


def main():
    """Función principal."""
    if len(sys.argv) < 2:
        print("Uso: python validate_paper_replication.py <notebook_path>")
        sys.exit(1)

    notebook_path = sys.argv[1]

    if not Path(notebook_path).exists():
        print(f"Error: Notebook no encontrado: {notebook_path}")
        sys.exit(1)

    validator = PaperValidator()
    results = validator.run_validation(notebook_path)

    # Exit code basado en compliance
    total = sum(len(checks) for checks in results.values())
    passed = sum(
        1 for checks in results.values() for c in checks if c["status"] == "PASS"
    )
    compliance_rate = (passed / total * 100) if total > 0 else 0

    sys.exit(0 if compliance_rate >= 80 else 1)


if __name__ == "__main__":
    main()
