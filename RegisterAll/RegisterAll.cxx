
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMaskImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkDirectory.h"
#include "itkFileTools.h"
#include "itkExceptionObject.h"
#include "itkCSVArray2DFileReader.h"

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

	if (argc < 5) {
		cout << "Please specify:" << endl;
		cout << " Input Directory" << endl;
		cout << " CSV File" << endl;
		cout << " Output Directory" << endl;
		cout << " Registration Program" << endl;
		cout << "[optional] fixed_image" << endl;
		return EXIT_FAILURE;
	} 

	double targets = 1;

	string input_dir_path(argv[1]);
	string csv_file(argv[2]);
	string output_dir_path(argv[3]);
	string registration(argv[4]);

	string output_tfm_dir_patha(output_dir_path + "/affine/transforms");
	string output_tfm_dir_pathb(output_dir_path + "/bspline/transforms");
	string output_img_dir_patha(output_dir_path + "/affine/images");
	string output_img_dir_pathb(output_dir_path + "/bspline/images");

	int pos;

	typedef itk::CSVArray2DFileReader<double> ReaderTypeCSV;
	ReaderTypeCSV::Pointer csv_reader = ReaderTypeCSV::New();

	csv_reader->SetFileName(csv_file);
	csv_reader->SetFieldDelimiterCharacter(';');
	csv_reader->SetStringDelimiterCharacter('"');
	csv_reader->HasColumnHeadersOn();
	csv_reader->HasRowHeadersOn();
	csv_reader->UseStringDelimiterCharacterOn();

	try {
		csv_reader->Parse();
	} catch (itk::ExceptionObject & e) {
		cerr << e << endl;
		return EXIT_FAILURE;
	}

	ReaderTypeCSV::Array2DDataObjectPointer data = csv_reader->GetOutput();


	Directory::Pointer input_dir = Directory::New();
	Directory::Pointer output_dir = Directory::New();

	Directory::Pointer output_img_dira = Directory::New();
	Directory::Pointer output_img_dirb = Directory::New();
	Directory::Pointer output_tfm_dira = Directory::New();
	Directory::Pointer output_tfm_dirb = Directory::New();

	try {
		input_dir->Load(input_dir_path.c_str());
		output_img_dira->Load(output_img_dir_patha.c_str());
		output_tfm_dira->Load(output_tfm_dir_patha.c_str());
		output_img_dirb->Load(output_img_dir_pathb.c_str());
		output_tfm_dirb->Load(output_tfm_dir_pathb.c_str());
	} catch (ExceptionObject & err) {
		cerr << err << endl;
		return EXIT_FAILURE;
	}	

	if (input_dir->GetNumberOfFiles() == 0) {
		cerr << "Input Directory does not exist" << endl;
		return EXIT_FAILURE;
	}

	if (output_img_dira->GetNumberOfFiles() == 0) {
		FileTools::CreateDirectoryA(output_img_dir_patha.c_str());
	}

	if (output_img_dirb->GetNumberOfFiles() == 0) {
		FileTools::CreateDirectoryA(output_img_dir_pathb.c_str());
	}

	if (output_tfm_dira->GetNumberOfFiles() == 0) {
		FileTools::CreateDirectoryA(output_tfm_dir_patha.c_str());
	}

	if (output_tfm_dirb->GetNumberOfFiles() == 0) {
		FileTools::CreateDirectoryA(output_tfm_dir_pathb.c_str());
	}

	int n = 0;
	for (int i = 0; i < data->GetRowHeaders().size(); ++i) {
		//cout << data->GetData(data->GetRowHeaders()[i], "Age") << endl;
		/*if (targets == data->GetData(data->GetRowHeaders()[i], "Status")) {
			++n;
		}*/
	}

	cout << data->GetData(data->GetRowHeaders()[0], data->GetColumnHeaders()[0]) << endl;

	cout << "n_images = " << n << endl;

	//return 0;

	string fixed_image_path;
	string transform_type = "affine";
	bool found_fixed = false;

	if (argc == 6) {
		fixed_image_path = argv[5];
		found_fixed = true;
	}

	for (int i = 0; i < data->GetRowHeaders().size(); ++i) {
		//cout << input_dir_path + "/" + data->GetRowHeaders()[i] + "/T1.nii.gz" << endl;
	
		//if (targets == data->GetData(data->GetRowHeaders()[i], "Status")) {
		if (!found_fixed) {
			fixed_image_path = input_dir_path + "/" + "BRCAD" + data->GetRowHeaders()[i] + "/T1.nii.gz";
			found_fixed = true;
		}

		string base_name = "BRCAD" + data->GetRowHeaders()[i];
		string moving_image_path = input_dir_path + "/" + base_name + "/T1.nii.gz";
		string output_img_path = output_img_dir_patha + "/" + base_name + ".nii";
		string output_tfm_path = output_tfm_dir_patha + "/" + base_name + ".tfm";

		string command;
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
		//cout << "command: " << command << endl;
		cout << "--" << endl;
		system(command.c_str());
		//}
	}

	transform_type = "bspline";
	for (int i = 0; i < data->GetRowHeaders().size(); ++i) {
		//cout << input_dir_path + "/" + data->GetRowHeaders()[i] + "/T1.nii.gz" << endl;
		//cout << (targets==data->GetData(data->GetRowHeaders()[i], "Status")) << endl;

		//if (targets == data->GetData(data->GetRowHeaders()[i], "Status")) {

		string base_name = "BRCAD" + data->GetRowHeaders()[i];
		string moving_image_path = output_img_dir_patha + "/" + base_name + ".nii";
		string output_img_path = output_img_dir_pathb + "/" + base_name + ".nii";
		string output_tfm_path = output_tfm_dir_pathb + "/" + base_name + ".tfm";

		string command;
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
		//}
	}


	return 0;

	//string command;
	//string file_path;
	//string file_name;
	//string output_img_path;
	//string output_tfm_path;	


	//Directory::Pointer test_file = Directory::New();

	//bool got_fixed_image = false;
	////string fixed_image_path;
	//string moving_image_path;

	//int counter = 0;

	//for (int i = 2; i < input_dir->GetNumberOfFiles(); ++i) {
	//	file_path = input_dir_path + "/" + input_dir->GetFile(i);
	//	test_file->Load(file_path.c_str());
	//	file_name = input_dir->GetFile(i);

	//	if (test_file->GetNumberOfFiles() > 0) continue;

	//	if (fixed_image_name == file_name) continue; 		

	//	moving_image_path = file_path;
	//	output_img_path = output_img_dir_path + "/" + output_img_name_short + to_string(counter) + output_img_name_ext;
	//	output_tfm_path = output_tfm_dir_path + "/" + output_tfm_name_short + to_string(counter) + output_tfm_name_ext;


	//	command = registration;
	//	command += " " + fixed_image_path;
	//	command += " " + moving_image_path;
	//	command += " " + output_img_path;
	//	command += " " + output_tfm_path;
	//	command += " " + transform_type;

	//	cout << "--" << endl;
	//	cout << "fixed_image: " << fixed_image_path << endl;
	//	cout << "moving_image: " << moving_image_path << endl;
	//	cout << "output_image: " << output_img_path << endl;
	//	cout << "output_transform: " << output_tfm_path << endl;
	//	cout << "transform_type: " << transform_type << endl;
	//	cout << "command: " << command << endl;
	//	cout << "--" << endl;

	//	system(command.c_str());
	//	++counter;
	//}




	return EXIT_SUCCESS;
}

