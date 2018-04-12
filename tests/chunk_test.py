import unittest
import pandas

from swnamer.process import chunk_names


class ChunkTests(unittest.TestCase):

    def setUp(self):
        self.test_df = pandas.DataFrame({'name': ['darth vader']})

    def test_single_timestep(self):
        chunks, next_chars = chunk_names(self.test_df, 'name', 1)
        self.assertEqual(['d', 'a', 'r', 't', 'h', ' ', 'v', 'a', 'd', 'e', 'r'], chunks)
        self.assertEqual(['a', 'r', 't', 'h', ' ', 'v', 'a', 'd', 'e', 'r', '\n'], next_chars)

    def test_two_timesteps(self):
        chunks, next_chars = chunk_names(self.test_df, 'name', 2)
        # 'darth vader'
        self.assertEqual(['da', 'ar', 'rt', 'th', 'h ', ' v', 'va', 'ad', 'de', 'er'], chunks)
        self.assertEqual(['r', 't', 'h', ' ', 'v', 'a', 'd', 'e', 'r', '\n'], next_chars)

    def test_three_timesteps(self):
        chunks, next_chars = chunk_names(self.test_df, 'name', 3)
        # 'darth vader'
        self.assertEqual(['dar', 'art', 'rth', 'th ', 'h v', ' va', 'vad', 'ade', 'der'], chunks)
        self.assertEqual(['t', 'h', ' ', 'v', 'a', 'd', 'e', 'r', '\n'], next_chars)
