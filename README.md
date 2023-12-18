# Souhaib
ChatbotPart1


# Importation des bibliothèques nécessaires
import os
import math

# Fonction pour obtenir la liste des fichiers avec une certaine extension dans un répertoire
def list_of_files(directory, extension):
    files_names = []
    # Parcourir tous les fichiers dans le répertoire
    for filename in os.listdir(directory):
        # Si le fichier a l'extension spécifiée, l'ajouter à la liste
        if filename.endswith(extension):
            files_names.append(filename)
    return files_names

# Fonction pour extraire le nom du président à partir du nom du fichier
def nom_president(files_names):
    liste_np = []
    for e in files_names:
        if "1"in e or "2" in e:
            liste_np.append(e[11 : -5])
        else :
            liste_np.append(e[11 : -4])
    return liste_np

# Fonction pour trouver le prénom du président à partir de la liste des noms
def trouve_prenom(lst_nom_prez):
    pre = ["Nicolas Sarkozy", "Emmanuel Macron","François Hollande", "Valéry Giscard dEstaing", "François Mitterrand", "Jacques Chirac"]
    res = []
    for e in lst_nom_prez:
        for i in pre:
            if e in i:
                res.append(i)
    return res

# Fonction pour supprimer les doublons d'une liste
def doublon(dou):
    res = []
    for e in dou:
        if e not in res:
            res.append(e)
    return res

# Fonction pour convertir le contenu d'un fichier en minuscules
def minus(files_names):
    for e in files_names:
        f = open("./speeches/" + e, "r")
        content = f.read()
        f.close()
        res = ''
        for i in content:
            if 65 <= ord(i) <= 90:
                res += chr(ord(i)+32)
            else:
                res += i
        if not os.path.exists('./cleaned'):
            os.mkdir('./cleaned')
        f = open("./cleaned/" + e, "w")
        f.write(res)
        f.close()

# Fonction pour supprimer la ponctuation d'un fichier
def ponctuation(files_names):
    for e in files_names:
        f = open("./cleaned/" + e, "r")
        contenu = f.read()
        f.close()
        str = ".;:?!,-()[]'"
        for i in str:
            contenu = contenu.replace(i,"")
        contenu = contenu.replace("-"," ")
        contenu = contenu.replace("'"," ")
        f = open("./cleaned/" + e, "w")
        f.write(contenu)
        f.close()

# Fonction pour compter le nombre d'occurrences de chaque mot dans un texte
def nb_occurence(o):
    dico = {}
    o = o.split()
    for i in o:
        if i not in dico:
            dico[i] = 1
        else:
            dico[i] += 1
    return dico

# Fonction pour calculer la fréquence des termes (TF) pour chaque fichier
def tf(files_names):
    matrix = {}
    for e in files_names:
        f = open('./cleaned/' + e, 'r')
        matrix[e] = nb_occurence(f.read())
        f.close()
    return matrix

# Fonction pour calculer l'IDF (Inverse Document Frequency) pour chaque mot dans chaque fichier
def compute_idf(word_matrix):
    idf_dict = {}
    word_files = {}
    n_files = len(word_matrix)

    for dico in word_matrix.values():
        for word in dico:
            word_files[word] = 0

    for dico in word_matrix.values():
        for word in dico:
            word_files[word] += 1

    for file_name, dico in word_matrix.items():
        idf_file = {}
        for word in dico:
            n_files_containing_word = word_files[word]
            idf_file[word] = math.log((n_files / n_files_containing_word) + 1)
        idf_dict[file_name] = idf_file

    return idf_dict

# Fonction pour calculer le score TF-IDF pour chaque mot dans chaque document
def compute_tf_idf(tf_dict, idf_dict):
    tf_idf_matrix = {}

    for file_name, tf_dico in tf_dict.items():
        tf_idf_dico = {}
        for word, freq_tf in tf_dico.items():
            idf_value = idf_dict[file_name][word]
            tf_idf = freq_tf * idf_value
            tf_idf_dico[word] = tf_idf
        tf_idf_matrix[file_name] = tf_idf_dico

    return tf_idf_matrix

# Fonction pour trouver les mots les moins importants dans chaque document
def find_least_important_words(tfidf_matrix):
    non_important_words = set()

    for file_name, tfidf_scores in tfidf_matrix.items():
        for word, score in tfidf_scores.items():
            if score == 0:
                non_important_words.add(word)

    return list(non_important_words)

# Fonction pour trouver les mots les plus importants dans chaque document
def find_most_important_words(tfidf_matrix):
    max_score = 0
    most_important_words = []

    for file_name, tfidf_scores in tfidf_matrix.items():
        for word, score in tfidf_scores.items():
            if score > max_score:
                max_score = score
                most_important_words = [word]
            elif score == max_score:
                most_important_words.append(word)

    return most_important_words

# Fonction pour trouver le(s) mot(s) plus répété(s)
def mots_plus_repetes(tfidf_results, president):
    max_score = 0
    mots_plus_repetes = []

    for file_name, tfidf_scores in tfidf_results.items():
        if president in file_name:
            for word, score in tfidf_scores.items():
                if score > max_score:
                    max_score = score
                    mots_plus_repetes = [word]
                elif score == max_score:
                    mots_plus_repetes.append(word)

    return mots_plus_repetes

# Fonction pour trouver le(s) nom(s) du (des) président(s) qui a (ont) parlé de la « Nation » et celui qui l’a répété le plus de
# fois
def president_parle_nation(tfidf_results, mot):
    president_mot_count = {}

    for file_name, tfidf_scores in tfidf_results.items():
        for word, score in tfidf_scores.items():
            if word == mot:
                president = file_name.split('_')[1]
                president = president.split('.')[0]
                if president in president_mot_count:
                    president_mot_count[president] += score
                else:
                    president_mot_count[president] = score

    president_max = max(president_mot_count, key=president_mot_count.get)

    return president_mot_count, president_max

# Fonction pour trouver les mots qui ont été dit par les présidents Hormis les mots dits « non importants »
def mots_tous_presidents(tfidf_results, non_important_words):
    all_words = {}

    for file_name, tfidf_scores in tfidf_results.items():
        for word, score in tfidf_scores.items():
            if word not in non_important_words:
                if word in all_words:
                    all_words[word] += 1
                else:
                    all_words[word] = 1

    mots_tous_presidents = []

    for word, count in all_words.items():
        if count == len(tfidf_results):
            mots_tous_presidents.append(word)

    return mots_tous_presidents

# Fonction pour trouver le premier président à parler du climat et/ou de l’écologie
def premier_president_parler_climat_ecologie(tfidf_results):
    files = (tfidf_results.keys())

    for file_name in files:
        tfidf_scores = tfidf_results[file_name]

        if "climat" in tfidf_scores or "écologie" in tfidf_scores:
            president = file_name.split('_')[1]
            president = president.split('.')[0]

            return president

    return None


# Call of the function
directory = "./speeches"
files_names = list_of_files(directory, "txt")
lst_nom_prez = nom_president(files_names)
dou = trouve_prenom(lst_nom_prez)
minus(files_names)
ponctuation(files_names)
tf_results = tf(files_names)
word_matrix = tf(files_names)
idf_results = compute_idf(word_matrix)
tfidf_results = compute_tf_idf(tf_results, idf_results)

#partie2
#fonction qui tokénise la question

import string

def tokenize_question(text):
    # Convertir le texte en minuscules
    text = text.lower()

    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Diviser le texte en mots
    words = text.split()

    return words

#fonction qui permet d'identifier les termes de la question qui sont également présents dans le corpus de documents

def find_question_terms_in_corpus(question_tokens, corpus_tfidf_results):
    # Créer un ensemble des mots uniques dans la question
    question_set = set(question_tokens)

    # Initialiser un ensemble pour stocker les mots de la question présents dans le corpus
    question_terms_in_corpus = set()

    # Parcourir les résultats TF-IDF du corpus
    for file_name, tfidf_scores in corpus_tfidf_results.items():
        # Créer un ensemble des mots uniques dans le fichier du corpus
        corpus_file_set = set(tfidf_scores.keys())

        # Trouver l'intersection entre l'ensemble des mots de la question et l'ensemble des mots du corpus
        intersection = question_set.intersection(corpus_file_set)

        # Ajouter les mots de l'intersection à l'ensemble des termes de la question présents dans le corpus
        question_terms_in_corpus.update(intersection)

    return question_terms_in_corpus

#--------------------------------------------------------------
question = ""
tokenize_question(question)

#--------------------------------------------------------------

#fonction qui calcule le vecteur TF-IDF pour les termes de la question

def calculate_question_tfidf_vector(question_tokens, corpus_tfidf_results):
    # Initialiser le vecteur TF-IDF de la question avec des zéros
    question_tfidf_vector = [0] * len(corpus_tfidf_results)

    # Trouver les termes de la question présents dans le corpus
    question_terms_in_corpus = find_question_terms_in_corpus(question_tokens, corpus_tfidf_results)

    # Parcourir les résultats TF-IDF du corpus
    for i, (file_name, tfidf_scores) in enumerate(corpus_tfidf_results.items()):
        # Parcourir les termes de la question
        for term in question_terms_in_corpus:
            # Associer à chaque mot de la question un score TF
            tf_score = question_tokens.count(term)
            # Utiliser le score IDF du terme du corpus
            idf_score = tfidf_scores.get(term, 0)
            # Calculer le score TF-IDF et l'ajouter au vecteur de la question
            question_tfidf_vector[i] += tf_score * idf_score

    return question_tfidf_vector

#fonctions pour effectuer le calcul de la similarité de cosinus entre le vecteur de la question et chaque vecteur de la matrice TF-IDF

import math

def dot_product(vector_a, vector_b):
    """Calcule le produit scalaire entre deux vecteurs."""
    return sum(a * b for a, b in zip(vector_a, vector_b))

def vector_norm(vector):
    """Calcule la norme d'un vecteur."""
    return math.sqrt(sum(a**2 for a in vector))

def cosine_similarity(vector_a, vector_b):
    """Calcule la similarité de cosinus entre deux vecteurs."""
    dot_prod = dot_product(vector_a, vector_b)
    norm_a = vector_norm(vector_a)
    norm_b = vector_norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0  # Éviter une division par zéro

    similarity = dot_prod / (norm_a * norm_b)
    return similarity

def find_most_similar_document(question_tfidf_vector, corpus_tfidf_matrix):
    """Trouve le document le plus similaire à la question."""
    most_similar_document = None
    max_similarity = -1

    for file_name, tfidf_vector in corpus_tfidf_matrix.items():
        similarity = cosine_similarity(question_tfidf_vector, tfidf_vector)

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_document = file_name

    return most_similar_document, max_similarity

#fonction pour calculer le document le plus pertinent

def find_most_relevant_document(question_tfidf_vector, corpus_tfidf_matrix, file_names):
    """Trouve le document le plus pertinent pour la question."""
    most_relevant_document = None
    max_similarity = -1

    for file_name, tfidf_vector in corpus_tfidf_matrix.items():
        similarity = cosine_similarity(question_tfidf_vector, tfidf_vector)

        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_document = file_name

    # Convertir le nom du fichier dans le répertoire "cleaned" vers le répertoire "speeches"
    most_relevant_document = convert_to_speeches_directory(most_relevant_document, file_names)

    return most_relevant_document, max_similarity

def convert_to_speeches_directory(cleaned_file_name, file_names):
    """Convertit le nom de fichier du répertoire "cleaned" vers le répertoire "speeches"."""
    # Extrayez le nom du président du fichier "cleaned"
    president_name = cleaned_file_name.split('_')[1].split('.')[0]

    # Recherche du fichier correspondant dans le répertoire "speeches"
    for file_name in file_names:
        if president_name in file_name:
            return file_name

    return None
    


def generate_response(question_tfidf_vector, most_relevant_document, corpus_texts):
    """Génère une réponse basée sur le mot avec le score TF-IDF le plus élevé."""
    # Trouver le mot avec le score TF-IDF le plus élevé dans la question
    max_tfidf_word = max(question_tfidf_vector, key=question_tfidf_vector.get)

    # Trouver la première occurrence du mot dans le document pertinent
    document_text = corpus_texts[most_relevant_document]
    start_index = document_text.find(max_tfidf_word)

    # Trouver la phrase qui contient le mot
    if start_index != -1:
        end_index = document_text.find('.', start_index)
        if end_index != -1:
            response = document_text[start_index:end_index + 1]
            return response.strip()

    return "Aucune réponse n'a été trouvée."
    
def refine_response(raw_response, question):
    """Affine la réponse en ajoutant une majuscule et un point, et en ajoutant des répliques basées sur la question."""
    # Ajouter une majuscule en début de phrase et un point à la fin
    refined_response = raw_response.capitalize().strip() + "."

    # Ajouter des répliques basées sur la forme de la question
    question_starters = {
        "Comment": "Après analyse, ",
        "Pourquoi": "Car, ",
        "Peux-tu": "Oui, bien sûr!",
        # Ajoutez d'autres formes de questions au besoin
    }

    # Trouver la forme de la question
    for starter, reply in question_starters.items():
        if question.startswith(starter):
            refined_response = reply + refined_response
            break

    return refined_response








