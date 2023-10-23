**Masterproject Vrije Universiteit Amsterdam**
Mickey van Immerseel

One way to find specific information efficiently in text is through information extraction. Recognizing named entities and concepts, such as genes and diseases, from text is the basis for most biomedical applications of information extraction. This method is called Named
Entity Recognition (NER). Two different methods of NER are dictionary-based Named Entity Recognition (d-NER) and Large Language Model-based NER. D-NER has been proven useful for the exploitation of specific information contained in scientific publications. Recently however, focus has been on using LLMs for this task. LLMs require a great amount of training in order to work effectively. They need large databases of annotated text and time. D-NERs are rule-based methods and only need predefined rules and a dictionary. However, a d-NER could still take a long time to run as this method has to compare almost all text to a dictionary. Another method of information extraction is semantic frame extraction using the Fluid Construction Grammar (FCG). The FCG applies a rule-based method to identify all instances of selected frames in text and could extract frames that are most likely to contain the needed information. Using specifically selected frames with the d-NER could increase the efficiency of the d-NER, by reducing bulk from text without reducing accuracy.
The present study will compare rule-based methods and LLM-based methods for disease and gene entity recognition from biomedical text on three dimensions (accuracy, F1-score and runtime). Additionally, we will research the effect of frame extraction on d-NER for the same dimensions.

**Explanation of the code**

The dictionary and gene tagger make use of information from Hetio.net. The data regarding the diseases can be found at Data/Disease/do_sparql_results(1).csv.  The gene2go file for the data regarding the genes was too large to commit through git. However, it can be found through https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/ 

The two taggers are described in Tagger.py. The dataframes needed to be edited to fit the function. If one wants to make use of different data for this test, make sure the format is the same as the currently used dataframes. The function uses both exact matching and fuzzy matching, so make sure that for the Gene Tagger fuzzy matching is commented out.

The LLM-based method is described in LLMSciBert.ipynb. The training and test data are now set to the Disease function, make sure to switch it out for the Gene function if one wants to run that. The training data and test data are available in their corresponding Data folders.
