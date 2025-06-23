import csv

def classCounter(hierarchy_csv):
    with open(hierarchy_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        NbClassesLevel = [0] * len(next(reader))
        KnownItems = []
        for row in reader:
            for i,col in enumerate(row):
                if col not in KnownItems:
                    NbClassesLevel[i] += 1
                    KnownItems.append(col)
    return NbClassesLevel