bucket_name = ''
output_folder = ''
model_file_name = ''
prefix = ''
model_key = ''


def set_input_folder_from_name(name):
    bucket_name = 'cicdlambdanew'
    output_folder = 'OutputImages/'
    model_file_name = 'mymodel_Dec13_keras_new_dataset.keras'
    prefix = 'Model_files'
    model_key = f'{prefix}/{model_file_name}'

    print(bucket_name,output_folder,model_file_name,prefix,model_key)
    



set_input_folder_from_name("wolf")
