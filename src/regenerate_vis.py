import casanova
from bertopic import BERTopic
from nltk.corpus import stopwords


DATAFILE = "mieux-sinformer.csv"
MODEL = "models/13-03-2023_bertopic.model"
DIR = "../docs/topic_visualisations/"

stoplist = stopwords.words("french")
ADDITIONAL_STOPWORDS = [
    "plus",
    "chaque",
    "tout",
    "tous",
    "toutes",
    "toute",
    "leur",
    "leurs",
    "comme",
    "afin",
    "pendant",
    "lorsque",
]
stoplist.extend(ADDITIONAL_STOPWORDS)


def get_docs(DATAFILE):
    with open(DATAFILE) as f:
        reader = casanova.reader(f)
        docs = [
            special_preprocessing(cell) for cell in reader.cells(column="Proposition")
        ]
        print(f"Dataset includes {len(docs)} docs.")
        return docs


def special_preprocessing(string):
    string = string.lower()
    bow = string.split()
    if bow[:2] == ["il", "faut"]:
        bow = bow[2:]
    return " ".join(bow)


def main():
    # LOAD MODEL
    topic_model = BERTopic.load(MODEL)

    # PARSE DOCUMENTS
    docs = get_docs(DATAFILE)

    # GENERATE MODEL WITH MERGED TOPICS
    topics_to_merge = [[3, 4, 6], [3, 15]]
    topic_model.merge_topics(docs, topics_to_merge)
    topics_to_merge = [[4, 12]]
    topic_model.merge_topics(docs, topics_to_merge)

    # RENAME TOPICS
    topic_labels_dict = {
        0: "L'opinion & le journalisme",
        1: "Financement & l'indépendance des médias",
        2: "Désinformation",
        3: "Formation secondaire",
        4: "Formation primaire",
        5: "Accès à l'information",
        6: "Chaînes d'information en continu",
        7: "Législation",
        8: "Éthique du journalisme",
        9: "Désanonymisation en ligne",
        10: "Arnaques & influenceurs",
        11: "Portée géographique des médias",
        12: "Enseignement & l'EMI",
    }
    topic_model.set_topic_labels(topic_labels_dict)

    # BAR CHART
    barchart = topic_model.visualize_barchart(
        top_n_topics=17,
        custom_labels=True,
        title="<b>Représentation des thèmes</b>",
        n_words=7,
        width=300,
        height=250,
    )
    outfile = DIR + "barchart.html"
    barchart.write_html(outfile)

    # HIERARCHY
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    hierarchy = topic_model.visualize_hierarchy(
        hierarchical_topics=hierarchical_topics,
        width=1200,
        custom_labels=topic_labels_dict,
    )
    outfile = DIR + "hierarchy.html"
    hierarchy.write_html(outfile)

    # HEATMAP
    heatmap = topic_model.visualize_heatmap(
        n_clusters=6, custom_labels=topic_labels_dict, width=1200
    )
    outfile = DIR + "heatmap.html"
    heatmap.write_html(outfile)


if __name__ == "__main__":
    main()
