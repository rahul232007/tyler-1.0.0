import unittest

from app.core.config import get_settings


class VoicePipelineConfigTests(unittest.TestCase):
    def test_voice_settings_are_available(self):
        settings = get_settings()
        self.assertTrue(hasattr(settings, "elevenlabs_voice_id"))
        self.assertTrue(hasattr(settings, "elevenlabs_model_id"))
        self.assertIsInstance(settings.elevenlabs_voice_id, str)
        self.assertIsInstance(settings.elevenlabs_model_id, str)


if __name__ == "__main__":
    unittest.main()
