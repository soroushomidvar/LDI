def print_list_with_suffix(lst):
    for i, item in enumerate(lst):
        if i == len(lst) - 1:
            print(f"{item} -> A")
        else:
            print(f"{item} -> A, ", end='')


# Example usage:
example_list = ['item1', 'item2', 'item3']
print_list_with_suffix(example_list)
