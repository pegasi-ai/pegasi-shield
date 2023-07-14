import unittest
from guardrail.metrics import textstat


class TestTextStat(unittest.TestCase):
    test_data = (
        "Playing games has always been thought to be important to "
        "the development of well-balanced and creative children; "
        "however, what part, if any, they should play in the lives "
        "of adults has never been researched that deeply. I believe "
        "that playing games is every bit as important for adults "
        "as for children. Not only is taking time out to play games "
        "with our children and other adults valuable to building "
        "interpersonal relationships but is also a wonderful way "
        "to release built up tension."
    )

    def test_automated_readability_index(self):
        self.assertEqual(textstat.aggregate_reading_level(self.test_data), 15.5)


unittest.main()
