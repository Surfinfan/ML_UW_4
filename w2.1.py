import sframe                            # see below for install instruction
import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix      # sparse matrices
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
%matplotlib inline


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)


def unpack_dict(matrix, map_index_to_word):
    table = list(map_index_to_word.sort('index')['category'])
    # if you're not using SFrame, replace this line with
    ##      table = sorted(map_index_to_word, key=map_index_to_word.get)


    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr

    num_doc = matrix.shape[0]

    return [{k: v for k, v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i + 1]]],
                                  data[indptr[i]:indptr[i + 1]].tolist())} \
            for i in xrange(num_doc)]


def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table.sort('count', ascending=False)


def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word','weight'])
    return word_count_table.sort('weight', ascending=False)


def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(word_count_vector.keys())
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return common_words.issubset(unique_words)


# El cuerpo de todos los datos de wikipedia sobre los que trabajaremos
wiki = sframe.SFrame('..//w2-a1//people_wiki.gl/')
wiki = wiki.add_row_number()

# El conteo de palabras de cada articulo nos lo dan, aunque se podría extraer por nuestros propios medios (explorar sklearn.CountVectorize)
word_count = load_sparse_csr('..//w2-a1//people_wiki_word_count.npz')
map_index_to_word = sframe.SFrame('..//w2-a1//people_wiki_map_index_to_word.gl/')

# Ahora experimentaremos con KNN. Primero con los word counts en Bruto
model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)

# Buscamos el artculo mas parecido a Obama
obama_id = wiki[wiki['name'] == 'Barack Obama']['id'][0]

distances, indices = model.kneighbors(word_count[obama_id], n_neighbors=10)
neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
print wiki.join(neighbors, on='id').sort('distance')[['id','name','distance']]


# Vamos a ver porque el KNN con word_counts en raw sale con gente tan 'rara' como parecida a Obama.

wiki['word_count'] = unpack_dict(word_count, map_index_to_word)

obama_words = top_words('Barack Obama')
print obama_words

barrio_words = top_words('Francisco Barrio')
print barrio_words

combined_words = obama_words.join(barrio_words, on='word')
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Barrio'})
combined_words.sort('Obama', ascending=False)

# Quiz Question: Cuantas entradas de la wikipedia tienen esas mismas 5 palabras mas comunes en su artículo?

common_words = set(combined_words.sort('Obama', ascending=False)[0:5]['word'])

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
print sum(wiki['has_top_words'])


# Quiz Question. Measure the pairwise distance between the Wikipedia pages of Barack Obama, George W. Bush, and Joe Biden.
# Which of the three pairs has the smallest distance?

obama_id = wiki[wiki['name'] == 'Barack Obama']['id'][0]
bush_id  = wiki[wiki['name'] == 'George W. Bush']['id'][0]
biden_id = wiki[wiki['name'] == 'Joe Biden']['id'][0]

obama_count = word_count[obama_id]
bush_count  = word_count[bush_id]
biden_count = word_count[biden_id]

obama_bush  = euclidean_distances(obama_count,bush_count)
obama_biden = euclidean_distances(obama_count,biden_count)
bush_biden  = euclidean_distances(bush_count,biden_count)

print "Distancia Obama-Bush: ", obama_bush
print "Distancia Obama-Biden: ", obama_biden
print "Distancia Bush-Biden: ", bush_biden

# Quiz Question. Collect all words that appear both in Barack Obama and George W. Bush pages.
# Out of those words, find the 10 words that show up most often in Obama's page.

obama_dict      = wiki[obama_id]['word_count']
bush_dict       = wiki[bush_id]['word_count']
obama_listwords = set(obama_dict.keys())
bush_listwords  = set(bush_dict.keys())

comunes = obama_listwords.intersection(bush_listwords)

obama_words.filter_by(list(comunes), 'word')

#---------------------------------------------------------------------------------------------------------------
# Ahora trabajaremos con TD-IDF

tf_idf = load_sparse_csr('..//w2-a1//people_wiki_tf_idf.npz')
wiki['tf_idf'] = unpack_dict(tf_idf, map_index_to_word)

# Nota: Nos dan el conteo TD-IDF, pero lo podriamos sacar nosotros.
# Mas info: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

# Creamos un nuevo modelo KNN para TD-IDF
model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)
distances, indices = model_tf_idf.kneighbors(tf_idf[obama_id], n_neighbors=10)

neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
print wiki.join(neighbors, on='id').sort('distance')[['id', 'name', 'distance']]

# Veamos como se parecen un par de articulos, el de Obanma y el de uno de sus colaboradores: Schiliro
obama_tf_idf = top_words_tf_idf('Barack Obama')
print obama_tf_idf

schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
print schiliro_tf_idf


#Quiz Question. Among the words that appear in both Barack Obama and Phil Schiliro, take the 5 that have largest
# weights in Obama. How many of the articles in the Wikipedia dataset contain all of those 5 words?

combined_words_td_idf = obama_tf_idf.join(schiliro_tf_idf, on='word')
combined_words_td_idf = combined_words_td_idf.rename({'weight':'Obama', 'weight.1':'Schiliro'})
combined_words_td_idf.sort('Obama', ascending=False)

common_words = set(combined_words_td_idf.sort('Obama', ascending=False)[0:5]['word'])

wiki['has_top_words_td_idf'] = wiki['tf_idf'].apply(has_top_words)

print sum(wiki['has_top_words_td_idf'])


# Quiz Question. Compute the Euclidean distance between TF-IDF features of Obama and Biden.

obama_count = tf_idf[obama_id]
biden_count = tf_idf[biden_id]

obama_biden = euclidean_distances(obama_count,biden_count)

print "Distancia Obama-Biden: ", obama_biden


# Biden está lejos porque penalizmaos documentos más largos sobre los cortos. Vamos a comporbarlo:

# Compute length of all documents
def compute_length(row):
    return len(row['text'].split(' '))
wiki['length'] = wiki.apply(compute_length)

# Compute 100 nearest neighbors and display their lengths
distances, indices = model_tf_idf.kneighbors(tf_idf[obama_id], n_neighbors=100)
neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
nearest_neighbors_euclidean = wiki.join(neighbors, on='id')[['id', 'name', 'length', 'distance']].sort('distance')
print nearest_neighbors_euclidean


# To see how these document lengths compare to the lengths of other documents in the corpus, let's make a histogram
# of the document lengths of Obama's 100 nearest neighbors and compare to a histogram of document lengths for all documents.

plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'][0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'][0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])

plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size':16})
plt.tight_layout()