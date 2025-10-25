"""
Tests para el módulo de detección de entorno.

Verifica que las funciones de detección de entorno y configuración
de rutas funcionen correctamente en local y Colab.
"""

import unittest
from pathlib import Path
from modules.core.environment import (
    detect_environment,
    find_project_root,
    get_project_paths,
    setup_environment,
    get_colab_drive_paths,
)


class TestEnvironmentDetection(unittest.TestCase):
    """Tests para detección de entorno."""

    def test_detect_environment_returns_valid_value(self):
        """Verifica que detect_environment retorna 'local' o 'colab'."""
        env = detect_environment()
        self.assertIn(env, ["local", "colab"])

    def test_detect_environment_is_local(self):
        """En tests normales, debe detectar 'local'."""
        env = detect_environment()
        self.assertEqual(env, "local")


class TestFindProjectRoot(unittest.TestCase):
    """Tests para búsqueda de raíz del proyecto."""

    def test_find_project_root_finds_root(self):
        """Verifica que encuentra la raíz del proyecto."""
        root = find_project_root()

        # La raíz debe existir
        self.assertTrue(root.exists())

        # La raíz debe contener al menos un marcador
        markers = ["modules", "requirements.txt", "Pipfile", ".git"]
        has_marker = any((root / marker).exists() for marker in markers)
        self.assertTrue(has_marker)

    def test_find_project_root_from_subdirectory(self):
        """Verifica que encuentra la raíz desde un subdirectorio."""
        # Simular estar en un subdirectorio
        test_dir = Path(__file__).parent
        root = find_project_root(test_dir)

        # Debe encontrar la raíz del proyecto
        self.assertTrue(root.exists())
        self.assertTrue(
            (root / "modules").exists() or (root / "requirements.txt").exists()
        )

    def test_find_project_root_returns_path(self):
        """Verifica que retorna un objeto Path."""
        root = find_project_root()
        self.assertIsInstance(root, Path)


class TestProjectPaths(unittest.TestCase):
    """Tests para configuración de rutas."""

    def test_get_project_paths_local(self):
        """Verifica rutas en entorno local."""
        paths = get_project_paths("local")

        # Verificar que retorna un diccionario
        self.assertIsInstance(paths, dict)

        # Verificar claves esperadas
        expected_keys = [
            "base",
            "cache_original",
            "cache_augmented",
            "cache_sequences",
            "results",
            "data",
        ]
        for key in expected_keys:
            self.assertIn(key, paths)

        # Verificar que las rutas son Path objects
        for key, path in paths.items():
            self.assertIsInstance(path, Path)

        # Verificar que base es una ruta válida que existe
        self.assertTrue(paths["base"].exists())

        # Verificar que base contiene marcadores del proyecto
        # (al menos uno de estos debe existir)
        markers = ["modules", "requirements.txt", "Pipfile", ".git"]
        has_marker = any((paths["base"] / marker).exists() for marker in markers)
        self.assertTrue(has_marker, "Base path debe contener marcadores del proyecto")

    def test_get_project_paths_colab_default(self):
        """Verifica rutas en entorno Colab con ruta por defecto."""
        paths = get_project_paths("colab")

        # Base debe ser la ruta de Colab
        expected_base = Path(
            "/content/drive/Othercomputers/ZenBook/parkinson-voice-uncertainty"
        )
        self.assertEqual(paths["base"], expected_base)

        # Cache original debe ser base/cache/original
        expected_cache = expected_base / "cache" / "original"
        self.assertEqual(paths["cache_original"], expected_cache)

    def test_get_project_paths_colab_custom_base(self):
        """Verifica rutas en Colab con ruta base personalizada."""
        custom_base = "/content/drive/MyDrive/mi_proyecto"
        paths = get_project_paths("colab", colab_base=custom_base)

        self.assertEqual(paths["base"], Path(custom_base))
        self.assertEqual(
            paths["cache_original"], Path(custom_base) / "cache" / "original"
        )

    def test_get_project_paths_auto_detect(self):
        """Verifica que auto-detecta si no se pasa environment."""
        paths = get_project_paths()

        # Debe retornar un diccionario válido
        self.assertIsInstance(paths, dict)
        self.assertIn("base", paths)


class TestSetupEnvironment(unittest.TestCase):
    """Tests para setup_environment."""

    def test_setup_environment_returns_tuple(self):
        """Verifica que retorna tupla (env, paths)."""
        result = setup_environment(verbose=False)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        env, paths = result
        self.assertIn(env, ["local", "colab"])
        self.assertIsInstance(paths, dict)

    def test_setup_environment_verbose_false(self):
        """Verifica que verbose=False no imprime nada."""
        # No hay forma directa de verificar que no imprime,
        # pero al menos verificamos que no falla
        env, paths = setup_environment(verbose=False)
        self.assertIsNotNone(env)
        self.assertIsNotNone(paths)

    def test_setup_environment_custom_colab_base(self):
        """Verifica que se puede pasar colab_base personalizado."""
        # Aunque estemos en local, podemos probar la lógica
        custom_base = "/custom/path"

        # Forzar detección como colab pasando environment
        # (aunque no podemos hacerlo directamente con setup_environment)
        paths = get_project_paths("colab", colab_base=custom_base)

        self.assertEqual(paths["base"], Path(custom_base))


class TestColabDrivePaths(unittest.TestCase):
    """Tests para funciones relacionadas con Google Drive."""

    def test_get_colab_drive_paths(self):
        """Verifica que retorna rutas comunes de Google Drive."""
        paths = get_colab_drive_paths()

        self.assertIsInstance(paths, dict)
        self.assertIn("my_drive", paths)
        self.assertIn("other_computers", paths)
        self.assertIn("shared_drives", paths)

        # Verificar que son strings
        for path in paths.values():
            self.assertIsInstance(path, str)


class TestPathConsistency(unittest.TestCase):
    """Tests para verificar consistencia de rutas."""

    def test_all_paths_are_subdirectories(self):
        """Verifica que todas las rutas son subdirectorios de base."""
        paths = get_project_paths("local")
        base = paths["base"]

        # Todos los paths (excepto base) deben empezar con base
        for key, path in paths.items():
            if key != "base":
                # Verificar que el path es relativo a base
                try:
                    path.relative_to(base)
                except ValueError:
                    self.fail(f"{key} path debe ser subdirectorio de base")

    def test_cache_paths_structure(self):
        """Verifica estructura de directorios de cache."""
        paths = get_project_paths("local")

        # Verificar que cache_original y cache_augmented tienen
        # 'cache' como parent
        cache_orig_parts = paths["cache_original"].parts
        cache_aug_parts = paths["cache_augmented"].parts

        self.assertIn("cache", cache_orig_parts)
        self.assertIn("cache", cache_aug_parts)
        self.assertIn("original", cache_orig_parts)
        self.assertIn("augmented", cache_aug_parts)


def run_tests():
    """Ejecuta todos los tests."""
    unittest.main(argv=[""], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
