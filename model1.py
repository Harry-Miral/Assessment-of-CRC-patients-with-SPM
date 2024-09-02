from pre_model1 import run_model1
run_num = 1
print('model1')

for i in range(3):
    year_num = i + 1
    text_name = 'h5model1/8/' + str(run_num) + 'fold_model1_alive_' + str(year_num) + '.txt'
    pth_name = 'h5model1/8/' + str(run_num) + 'fold_model1_alive_' + str(year_num)
    run_model1(year_num, text_name, pth_name)
    print('run_num:', run_num, 'alive:', year_num)

print('finish!')

