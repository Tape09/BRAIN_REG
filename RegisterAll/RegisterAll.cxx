
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMaskImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkDirectory.h"
#include "itkFileTools.h"
#include "itkExceptionObject.h"


#include <iostream>
#include <string>


typedef itk::Image< short, 3 > ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter< ImageType  > WriterType;
typedef itk::MaskImageFilter<ImageType, ImageType, ImageType> MaskFilterType;
typedef itk::ImageDuplicator< ImageType > DuplicatorType;


using namespace std;
using namespace itk;

int main(int argc, char *argv[]) {

	if (argc < 7) {
		cout << "Please specify:" << endl;
		cout << " Input Directory" << endl;
		cout << " Output Directory" << endl;
		cout << " Fixed Image Name" << endl;
		cout << " Output Name" << endl;
		cout << " Transform Type" << endl;
		cout << " Registration Program" << endl;
		return EXIT_FAILURE;
	} 

	string input_dir_path(argv[1]);
	string output_dir_path(argv[2]);
	string fixed_image_path(argv[3]);
	string output_img_name(argv[4]);
	output_img_name += ".nii";
	string output_tfm_name(argv[4]);
	output_tfm_name += ".tfm";
	string transform_type(argv[5]);
	string registration(argv[6]);

	string output_tfm_dir_path(output_dir_path + "/transforms");
	string output_img_dir_path(output_dir_path + "/images");

	int pos;

	
	string output_img_name_short = argv[4];
	string output_img_name_ext = ".nii";
	
	string output_tfm_name_short = argv[4];
	string output_tfm_name_ext = ".tfm";

	pos = fixed_image_path.find_last_of("/");
	string fixed_image_name = fixed_image_path.substr(pos+1);

	Directory::Pointer input_dir = Directory::New();
	Directory::Pointer output_dir = Directory::New();

	Directory::Pointer output_img_dir = Directory::New();
	Directory::Pointer output_tfm_dir = Directory::New();

	try {
		input_dir->Load(input_dir_path.c_str());
		output_img_dir->Load(output_img_dir_path.c_str());
		output_tfm_dir->Load(output_tfm_dir_path.c_str());
	} catch (ExceptionObject & err) {
		cerr << err << endl;
		return EXIT_FAILURE;
	}	

	if (input_dir->GetNumberOfFiles() == 0) {
		cerr << "Input Directory does not exist" << endl;
		return EXIT_FAILURE;
	}

	if (output_img_dir->GetNumberOfFiles() == 0) {
		FileTools::CreateDirectoryA(output_img_dir_path.c_str());
	}

	if (output_tfm_dir->GetNumberOfFiles() == 0) {
		FileTools::CreateDirectoryA(output_tfm_dir_path.c_str());
	}

	string command;
	string file_path;
	string file_name;
	string output_img_path;
	string output_tfm_path;


	Directory::Pointer test_file = Directory::New();

	bool got_fixed_image = false;
	//string fixed_image_path;
	string moving_image_path;

	int counter = 0;

	for (int i = 2; i < input_dir->GetNumberOfFiles(); ++i) {
		file_path = input_dir_path + "/" + input_dir->GetFile(i);
		test_file->Load(file_path.c_str());
		file_name = input_dir->GetFile(i);

		if (test_file->GetNumberOfFiles() > 0) continue;

		if (fixed_image_name == file_name) continue; 		

		moving_image_path = file_path;
		output_img_path = output_img_dir_path + "/" + output_img_name_short + to_string(counter) + output_img_name_ext;
		output_tfm_path = output_tfm_dir_path + "/" + output_tfm_name_short + to_string(counter) + output_tfm_name_ext;


		command = registration;
		command += " " + fixed_image_path;
		command += " " + moving_image_path;
		command += " " + output_img_path;
		command += " " + output_tfm_path;
		command += " " + transform_type;

		cout << "--" << endl;
		cout << "fixed_image: " << fixed_image_path << endl;
		cout << "moving_image: " << moving_image_path << endl;
		cout << "output_image: " << output_img_path << endl;
		cout << "output_transform: " << output_tfm_path << endl;
		cout << "transform_type: " << transform_type << endl;
		cout << "command: " << command << endl;
		cout << "--" << endl;

		system(command.c_str());
		++counter;
	}




	return EXIT_SUCCESS;
}

