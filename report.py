#draw box plot, write text
import seaborn as sns
%matplotlib inline
plt.figure(figsize=(10,8))
ax=sns.boxplot(x='id',y='fid_score',data=df,palette='rainbow')
plt.title('PGGAN lr 0.0001', fontsize=20)
plt.xlabel('ID', fontsize=20)
plt.ylabel('FID score', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
#plt.text(2, 340, '1_007400 '+ r'$\mu=$'+ str("%.2f" % np.mean(fid_arr_0)) + ' ' r'$\sigma=$' + str("%.2f" % np.std(fid_arr_0)), fontsize=13, color='blue')
#plt.text(2, 328, '2_010101 '+ r'$\mu=$'+ str("%.2f" % np.mean(fid_arr_1)) + ' ' r'$\sigma=$' + str("%.2f" % np.std(fid_arr_1)), fontsize=13, color='cyan')
#plt.text(2, 316, '3_012100 '+ r'$\mu=$'+ str("%.2f" % np.mean(fid_arr_2)) + ' ' r'$\sigma=$' + str("%.2f" % np.std(fid_arr_2)), fontsize=13, color='green')
#plt.text(2, 304, '4_015000 '+ r'$\mu=$'+ str("%.2f" % np.mean(fid_arr_3)) + ' ' r'$\sigma=$' + str("%.2f" % np.std(fid_arr_3)), fontsize=13, color='orange')
plt.text(2.3, 340, r'$\mu=$'+ str("%.2f" % np.mean(fid_arr_0)) + ' ' r'$\sigma=$' + str("%.2f" % np.std(fid_arr_0)), fontsize=15, color='blue')
plt.text(2.3, 323, r'$\mu=$'+ str("%.2f" % np.mean(fid_arr_1)) + ' ' r'$\sigma=$' + str("%.2f" % np.std(fid_arr_1)), fontsize=15, color='cyan')
plt.text(2.3, 306, r'$\mu=$'+ str("%.2f" % np.mean(fid_arr_2)) + ' ' r'$\sigma=$' + str("%.2f" % np.std(fid_arr_2)), fontsize=15, color='green')
plt.text(2.3, 289, r'$\mu=$'+ str("%.2f" % np.mean(fid_arr_3)) + ' ' r'$\sigma=$' + str("%.2f" % np.std(fid_arr_3)), fontsize=15, color='orange')
plt.show()
