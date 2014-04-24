
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

	if (argc < 6) {
		cout << "Please specify:" << endl;
		cout << " Input Directory" << endl;
		cout << " Output Directory" << endl;
		cout << " Output Image Name" << endl;
		cout << " Output Transform Name" << endl;
		cout << " Transform Type" << endl;
		return EXIT_FAILURE;
	} 

	string input_dir_path(argv[1]);
	string output_dir_path(argv[2]);
	string output_img_name(argv[3]);
	string output_tfm_name(argv[4]);
	string transform_type(argv[5]);

	string output_tfm_dir_path(output_dir_path + "/transforms");
	string output_img_dir_path(output_dir_path + "/images");

	int pos;

	pos = output_img_name.find_first_of(".");
	string output_img_name_short = output_img_name.substr(0, pos);
	string output_img_name_ext = output_img_name.substr(pos);

	pos = output_tfm_name.find_first_of(".");
	string output_tfm_name_short = output_tfm_name.substr(0, pos);
	string output_tfm_name_ext = output_tfm_name.substr(pos);

	string registration;

	if (argc == 7) {
		registration = argv[6];
	} else {
		registration = "test.exe";
	}

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
	string output_img_path;
	string output_tfm_path;


	Directory::Pointer test_file = Directory::New();

	bool got_fixed_image = false;
	string fixed_image_path;
	string moving_image_path;

	int counter = 0;

	for (int i = 2; i < input_dir->GetNumberOfFiles(); ++i) {
		file_path = input_dir_path + "/" + input_dir->GetFile(i);
		test_file->Load(file_path.c_str());

		if (test_file->GetNumberOfFiles() > 0) continue;

		if (!got_fixed_image) {
			fixed_image_path = file_path;
			got_fixed_image = true;
			continue;
		}

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
		cout << "--" << endl;

		system(command.c_str());
		++counter;
	}




	return EXIT_SUCCESS;
}

