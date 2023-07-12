import logging
from textstat import textstat

class Textstat:
    def __init__(self):
        self.ts = textstat
        self.logger = logging.getLogger("Textstat")
        logging.basicConfig(level=logging.INFO)

    def evaluate(self, text):
        metrics = {
            "automated_readability_index": {
                "col_type": "String",
                "schema_name": None,
                "function": lambda text: str(self.ts.automated_readability_index(text))
            },
            "dale_chall_readability_score": {
                "col_type": "String",
                "schema_name": "text_standard_component",
                "function": lambda text: str(self.ts.dale_chall_readability_score(text))
            },
            "linsear_write_formula": {
                "col_type": "String",
                "schema_name": "text_standard_component",
                "function": lambda text: str(self.ts.linsear_write_formula(text))
            },
            "gunning_fog": {
                "col_type": "String",
                "schema_name": "text_standard_component",
                "function": lambda text: str(self.ts.gunning_fog(text))
            },
            "aggregate_reading_level": {
                "col_type": "String",
                "schema_name": None,
                "function": lambda text: str(self.ts.text_standard(text, float_output=True))
            },
            "fernandez_huerta": {
                "col_type": "String",
                "schema_name": "es",
                "function": lambda text: str(self.ts.fernandez_huerta(text))
            },
            "szigriszt_pazos": {
                "col_type": "String",
                "schema_name": "es",
                "function": lambda text: str(self.ts.szigriszt_pazos(text))
            },
            "gutierrez_polini": {
                "col_type": "String",
                "schema_name": "es",
                "function": lambda text: str(self.ts.gutierrez_polini(text))
            },
            "crawford": {
                "col_type": "String",
                "schema_name": "es",
                "function": lambda text: str(self.ts.crawford(text))
            },
            "gulpease_index": {
                "col_type": "String",
                "schema_name": "it",
                "function": lambda text: str(self.ts.gulpease_index(text))
            },
            "osman": {
                "col_type": "String",
                "schema_name": "ar",
                "function": lambda text: str(self.ts.osman(text))
            },
            "flesch_kincaid_grade": {
                "col_type": "String",
                "schema_name": "text_standard_component",
                "function": lambda text: str(self.ts.flesch_kincaid_grade(text))
            },
            "flesch_reading_ease": {
                "col_type": "String",
                "schema_name": None,
                "function": lambda text: str(self.ts.flesch_reading_ease(text))
            },
            "smog_index": {
                "col_type": "String",
                "schema_name": "text_standard_component",
                "function": lambda text: str(self.ts.smog_index(text))
            },
            "coleman_liau_index": {
                "col_type": "String",
                "schema_name": "text_standard_component",
                "function": lambda text: str(self.ts.coleman_liau_index(text))
            },
            "sentence_count": {
                "col_type": "String",
                "schema_name": None,
                "function": lambda text: str(self.ts.sentence_count(text))
            },
            "character_count": {
                "col_type": "String",
                "schema_name": None,
                "function": lambda text: str(self.ts.char_count(text))
            },
            "letter_count": {
                "col_type": "String",
                "schema_name": None,
                "function": lambda text: str(self.ts.letter_count(text))
            },
            "polysyllable_count": {
                "col_type": "String",
                "schema_name": None,
                "function": lambda text: str(self.ts.polysyllabcount(text))
            },
            "monosyllable_count": {
                "col_type": "String",
                "schema_name": None,
                "function": lambda text: str(self.ts.monosyllabcount(text))
            },
            "difficult_words": {
                "col_type": "String",
                "schema_name": None,
                "function": lambda text: str(self.ts.difficult_words(text))
            },
            "syllable_count": {
                "col_type": "String",
                "schema_name": None,
                "function": lambda text: str(self.ts.syllable_count(str(text)))
            },
            "lexicon_count": {
                "col_type": "String",
                "schema_name": None,
                "function": lambda text: str(self.ts.lexicon_count(text))
            }
        }
        
        results = {}
        for metric, config in metrics.items():
            try:
                result = config['function'](text)
                self.logger.info(f"{metric}: {result}")
                results[metric] = result
            except Exception as e:
                self.logger.error(f"Error while evaluating {metric}: {e}")
                results[metric] = None
        
        return results
