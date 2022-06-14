def tree_analysis(trees):
    """
    Counts the occurrencies of terminals and primitives in the trees
    :param trees: trees to analyze
    :return:
    """
    # Creates dictionaries to count occurrences
    terminal_occurrences = dict()
    primitive_occurrences = dict()

    def get_trees(path):
        """
        Get the trees into a result file
        :param path: path of the file
        :return: trees
        """
        trees = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.split("; TREE: ")
                if len(line) > 1: trees.append(line[1])
        return trees

    # Argument is a path
    if isinstance(trees, str):
        trees = get_trees(trees)

    for tree in trees:
        tree = str(tree)
        tree = tree.replace("(", " ")
        tree = tree.replace(")", " ")
        tree = tree.replace(",", " ")

        tree = tree.split()

        # For each tree count terminals and primitives
        for i in range(len(tree)):
            elem = tree[i]
            if elem[:3] == "ARG": #(isinstance(tree[i], gp.Terminal)):
                if elem not in terminal_occurrences:
                    terminal_occurrences[elem] = 1
                else:
                    terminal_occurrences[elem] += 1
            else: # It's primitive
                if elem not in primitive_occurrences:
                    primitive_occurrences[elem] = 1
                else:
                    primitive_occurrences[elem] += 1
    # Get a view of the occurrencies
    terminal_view = [(v, k) for k, v in terminal_occurrences.items()]
    primitive_view = [(v, k) for k, v in primitive_occurrences.items()]

    terminal_view.sort(reverse=True)
    primitive_view.sort(reverse=True)

    print("=======================")
    print("TERMINALS OCCURRENCES: ")
    for v, k in terminal_view:
        print(f"{k}: {v}")
    print()
    print("PRIMITIVE OCCURRENCES: ")
    for v, k in primitive_view:
        print(f"{k}: {v}")
    print("=======================")

if __name__ == "__main__":
    # Analyze the results of a specific file
    FILENAME = "results_18.txt"
    tree_analysis("output/" + FILENAME)


