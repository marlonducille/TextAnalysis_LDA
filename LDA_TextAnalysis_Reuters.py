
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

from nltk.corpus import reuters






#print(reuters.words())
def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents");
 
    train_docs = list(filter(lambda doc: doc.startswith("train"),
                        documents));
    print(str(len(train_docs)) + " total train documents");
 
    test_docs = list(filter(lambda doc: doc.startswith("test"),
                       documents));
    print(str(len(test_docs)) + " total test documents");
 
    # List of categories
    categories = reuters.categories();
    print(categories)
    print(str(len(categories)) + " categories");
 
    # Documents in a category
    category_docs = reuters.fileids("acq");
 
    # Words for a document
    document_id = category_docs[0]
    document_words = reuters.words(category_docs[0]);
    print(document_words);  
 
    # Raw document
    print(reuters.raw(document_id));
    
    
collection_stats()
    






