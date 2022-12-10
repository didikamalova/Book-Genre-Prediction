import numpy as np
import torch
from utils import load_labels, load_dataset, evaluate, analyze_class_accuracy, get_confusion_matrix

# temp_dir = 'final_outputs/temp.csv'
# temp_out = torch.from_numpy(load_dataset(temp_dir))

title_output_dir = 'final_outputs/test_title_output.csv'
image_output_dir = 'final_outputs/test_image_output.csv'
comb2_output_dir = 'final_outputs/test_comb2_output.csv'
comb3_output_dir = 'final_outputs/test_comb3_output.csv'
test_y_dir = 'final_outputs/test_labels.csv'

title_out = torch.from_numpy(load_dataset(title_output_dir))
image_out = torch.from_numpy(load_dataset(image_output_dir))
comb2_out = torch.from_numpy(load_dataset(comb2_output_dir))
comb3_out = torch.from_numpy(load_dataset(comb3_output_dir))
test_y = torch.from_numpy(load_labels(test_y_dir)).type(torch.LongTensor)


def accuracies(filename):
    evaluate(image_out, test_y, "test")
    _, i_accuracy1 = analyze_class_accuracy(image_out, test_y, 1)
    _, i_accuracy3 = analyze_class_accuracy(image_out, test_y, 3)
    evaluate(title_out, test_y, "test")
    _, t_accuracy1 = analyze_class_accuracy(title_out, test_y, 1)
    _, t_accuracy3 = analyze_class_accuracy(title_out, test_y, 3)
    evaluate(comb2_out, test_y, "test")
    _, c2_accuracy1 = analyze_class_accuracy(comb2_out, test_y, 1)
    _, c2_accuracy3 = analyze_class_accuracy(comb2_out, test_y, 3)
    evaluate(comb3_out, test_y, "test")
    _, c3_accuracy1 = analyze_class_accuracy(comb3_out, test_y, 1)
    _, c3_accuracy3 = analyze_class_accuracy(comb3_out, test_y, 3)

    final_accuracy = np.array([i_accuracy1, i_accuracy3, t_accuracy1, t_accuracy3,
                               c2_accuracy1, c2_accuracy3, c3_accuracy1, c3_accuracy3]).T
    print(final_accuracy.shape)
    np.savetxt('./writeup_data/' + filename, final_accuracy, delimiter=',')


def conf_mats(output, y_true, filename):
    get_confusion_matrix(output, y_true, './writeup_data/' + filename)


accuracies('final_accuracies.csv')

conf_mats(title_out, test_y, 'title_conf')
conf_mats(image_out, test_y, 'image_conf')
conf_mats(comb2_out, test_y, 'comb2_conf')
conf_mats(comb3_out, test_y, 'comb3_conf')
# conf_mats(temp_out, test_y, 'temp_conf')
