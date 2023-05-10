import csv
from statistics import mean

import casanova
import networkx as nx
import pelote
import rich.progress


def categorize_age(age):
    try:
        age = int(age)
        if age < 20:
            return age, "moins de 20", "Génération Z"
        elif age < 30:
            if age < 26:
                return age, "20-29", "Génération Z"
            else:
                return age, "20-29", "Millénaire"
        elif age < 40:
            return age, "30-39", "Millénaire"
        elif age < 50:
            return age, "40-49", "Génération X"
        elif age < 60:
            return age, "50-59", "Génération X"
        elif age < 70:
            return age, "60-69", "Baby boomer"
        elif age < 80:
            return age, "70-79", "Baby boomer"
        elif age < 90:
            return age, "80-89", "Génération silencieuse"
        else:
            return None, "null", "null"
    except:
        return None, "null", "null"


def make_edges(agreement):
    PREDICTIONS_FILE = "bertopic_topics_enumerated.csv"

    with open(PREDICTIONS_FILE) as f:
        reader = csv.DictReader(f)
        rows = []
        from collections import Counter

        counter = Counter()
        for row in reader:
            age = row["Âge"]
            age_integer, age_range, age_category = categorize_age(age)
            if age_integer:
                row["age_integer"] = age_integer
            row["age_range"] = age_range
            row["Géneration de l'auteur de la proposition"] = age_category
            row["index"] = int(row["index"])
            counter.update([age_category])
            rows.append(row)
        doc_index = {row["Id"]: row for row in rows}
        print(counter)

    MATRIX_FILE = "defacto_covotes.csv"

    count = casanova.reader.count(MATRIX_FILE)
    with open(MATRIX_FILE) as f:
        reader = casanova.reader(f)
        pid1_pos = reader.headers["pid1"]
        pid2_pos = reader.headers["pid2"]
        vote1 = reader.headers["vote1"]
        vote2 = reader.headers["vote2"]
        count_pos = reader.headers["count"]
        total_count_pos = reader.headers["cocounts"]

        G = nx.Graph()
        for row in rich.progress.track(reader, total=count):
            covotoes_proposition1_id = row[pid1_pos]
            covotoes_proposition2_id = row[pid2_pos]

            # If the matrix refers to documents not in the original data, skip
            if not doc_index.get(covotoes_proposition1_id) or not doc_index.get(
                covotoes_proposition2_id
            ):
                continue

            # Base an edge's weight on the support that two related propositions received
            average_nb_votes = mean(
                [
                    int(doc_index[covotoes_proposition1_id]["Nb de votes"]),
                    int(doc_index[covotoes_proposition2_id]["Nb de votes"]),
                ]
            )
            weight = int(row[count_pos]) / average_nb_votes
            # weight = int(row[count_pos])

            # Unless already added to the Graph, add both nodes in the matrix row and create an edge between them
            if not G.has_node(covotoes_proposition1_id):
                G.add_node(
                    covotoes_proposition1_id,
                    label=doc_index[covotoes_proposition1_id]["Proposition"],
                    **doc_index[covotoes_proposition1_id],
                )

            if not G.has_node(covotoes_proposition2_id):
                G.add_node(
                    covotoes_proposition2_id,
                    label=doc_index[covotoes_proposition2_id]["Proposition"],
                    **doc_index[covotoes_proposition2_id],
                )

            # Create an edge if the two propositions received the same type of vote
            if agreement:
                if row[vote1] == row[vote2]:
                    G.add_edge(
                        covotoes_proposition1_id,
                        covotoes_proposition2_id,
                        weight=weight,
                    )
            else:
                if row[vote1] != row[vote2]:
                    G.add_edge(
                        covotoes_proposition1_id,
                        covotoes_proposition2_id,
                        weight=weight,
                    )
    return G


def create_gexf(G, name):
    GEFX_FILE = f"{name}.gexf"

    H = pelote.multiscale_backbone(G, alpha=0.05)
    nx.write_gexf(H, GEFX_FILE)


if __name__ == "__main__":
    G = make_edges(agreement=True)
    create_gexf(G, "agreement")
