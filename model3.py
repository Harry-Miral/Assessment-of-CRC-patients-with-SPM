from pre_model3 import run_model3

run_num = 1


year_num = 1
text_name = 'h5model3/new/5/' + str(run_num) + 'fold_model1_alive_' + str(year_num) + '.txt'
pth_name = 'h5model3/new/5/' + str(run_num) + 'fold_model1_alive_' + str(year_num)
run_model3(year_num, text_name, pth_name)
print('run_num:', run_num, 'alive:', year_num)

year_num = 2
text_name = 'h5model3/new/5/' + str(run_num) + 'fold_model1_alive_' + str(year_num) + '.txt'
pth_name = 'h5model3/new/5/' + str(run_num) + 'fold_model1_alive_' + str(year_num)
run_model3(year_num, text_name, pth_name)
print('run_num:', run_num, 'alive:', year_num)

year_num = 3
text_name = 'h5model3/new/5/' + str(run_num) + 'fold_model1_alive_' + str(year_num) + '.txt'
pth_name = 'h5model3/new/5/' + str(run_num) + 'fold_model1_alive_' + str(year_num)
run_model3(year_num, text_name, pth_name)
print('run_num:', run_num, 'alive:', year_num)

print('finish!')
