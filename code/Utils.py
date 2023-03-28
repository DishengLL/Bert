# print variables in the list
def print_list(var_list):
    for i, var_value in enumerate(var_list):
        var_name = f'my_list[{i}]'
        print(f'{var_name} = {var_value}')