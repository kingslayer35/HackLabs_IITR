import unittest

from .preprocessing import Preprocessor, clean_mention, context_diagnostics


class PreprocessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preprocessor=Preprocessor([
            'MS - Multiple sclerosis', 'MS - Mitral stenosis',
            'HTN - Hypertension', 'II - Implant insertion',
            'HEAL - Hysteroscopic endometrial ablation by laser',
        ]+['chronic kidney disease without gangrene stage left right no type diabetes heal']*8
          +['weaver hoarseness desaturase cardiomyopathy preexisting']*8)

    def test_formatting_preserves_clinical_qualifiers(self):
        self.assertEqual(clean_mention('header_4: &lt;b&gt;No LEFT type 2 diabetes&lt;/b&gt; (HCC)'),
                         'no left type 2 diabetes')

    def test_raw_is_retained_and_ambiguous_senses_stay_separate(self):
        result=self.preprocessor.views('MS')
        self.assertEqual(result['views'][0],dict(kind='raw',text='MS'))
        self.assertEqual(set(result['expansions'][0]['choices']),{'multiple sclerosis','mitral stenosis'})

    def test_does_not_expand_roman_numerals_or_ordinary_words(self):
        for mention in ('obesity class ii','slow to heal ulcers','no left type 2 diabetes'):
            with self.subTest(mention=mention):
                self.assertEqual(self.preprocessor.views(mention)['expansions'],[])

    def test_rejects_false_spelling_changes(self):
        for word in ('weaker','coarseness','desaturate'):
            with self.subTest(word=word):
                self.assertEqual(self.preprocessor.repair_word(word),(word,None))

    def test_recovers_typos_and_merged_words(self):
        for source,target in [('cardiomyopahty','cardiomyopathy'),('prexisting','preexisting'),
                              ('chronickidney','chronic kidney'),('withoutgangrene','without gangrene')]:
            with self.subTest(source=source):
                self.assertEqual(self.preprocessor.repair_word(source)[0],target)

    def test_context_is_marked_without_rewriting(self):
        self.assertEqual(context_diagnostics('pain','no pain today')['window'],'no [M]pain[/M] today')
        self.assertTrue(context_diagnostics('pain','')['no_context'])
        self.assertFalse(context_diagnostics('pain','other chart text')['mention_found'])


if __name__=='__main__':unittest.main()
