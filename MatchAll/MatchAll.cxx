

#include "itkCSVArray2DFileReader.h"
#include "itkCSVNumericObjectFileWriter.h"
#include "itkDirectory.h"
#include "itkFileTools.h"
#include "itkExceptionObject.h"
#include "itkTimeProbe.h"

#include <fstream>
#include <iostream>
#include <string>





//typedef itk::ImageFileWriter< ImageTypePCAtest > WriterTypePCAtest;



using namespace std;
using namespace itk;


int main(int argc, char *argv[]) {
	itk::TimeProbe totalTimeClock;
	totalTimeClock.Start();

	if (argc < 7) {
		cout << "Please specify:" << endl;
		cout << " CSV File" << endl;
		cout << " Input folder" << endl;
		cout << " Temp Folder" << endl;
		cout << " Output Path" << endl;
		cout << " PCA.exe" << endl;
		//cout << " sigpix.m" << endl;
		cout << " MatchModel.exe" << endl;
		cout << " [STOP AT]" << endl;
		
		//cout << " [Reconstruct]" << endl;
		return EXIT_FAILURE;
	}

	double pval_thresh = 0.01;

	//string asdf = "start ../Predict.exe";
	//system(asdf.c_str());
	//return 0;

	typedef CSVNumericObjectFileWriter<double> WriterTypeCSV;
	typedef itk::CSVArray2DFileReader<double> ReaderTypeCSV;

	string csv_path(argv[1]);
	string input_img_dir_path(argv[2]);
	string temp_path(argv[3]);
	string output_path(argv[4]);
	string pca_exe(argv[5]);
	//string sigpix_m(argv[6]);
	//sigpix_m = sigpix_m.substr(0, sigpix_m.size() - 2);
	string match_exe(argv[6]);
	string sp = " ";
	string start = "start";

	ReaderTypeCSV::Pointer csv_reader = ReaderTypeCSV::New();

	csv_reader->SetFileName(csv_path);
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

	vector<string> prop_names;
	vector<vector<double>> prop_vals;
	string fn_prefix = "BRCAD";
	string fn_img_suffix = ".nii";
	string fn_tfm_suffix = ".tfm";

	for (int i = 0; i < data->GetColumnHeaders().size(); ++i) {
		prop_names.push_back(data->GetColumnHeaders()[i]);
	}

	vector<string> img_names;
	for (int i = 0; i < data->GetRowHeaders().size(); ++i) {
		img_names.push_back(data->GetRowHeaders()[i]);
	}

	Directory::Pointer output_dir = Directory::New();
	try {
		output_dir->Load(output_path.c_str());
	} catch (ExceptionObject & err) {
		cerr << err << endl;
		return EXIT_FAILURE;
	}

	if (output_dir->GetNumberOfFiles() == 0) {
		FileTools::CreateDirectoryA(output_path.c_str());
	}

	Directory::Pointer temp_dir = Directory::New();
	try {
		temp_dir->Load(temp_path.c_str());
	} catch (ExceptionObject & err) {
		cerr << err << endl;
		return EXIT_FAILURE;
	}

	if (temp_dir->GetNumberOfFiles() == 0) {
		FileTools::CreateDirectoryA(temp_path.c_str());
	}

	string stop_at;
	if (argc >= 8) {
		stop_at = argv[7];		
	} else {
		stop_at = img_names.back();
	}

	//string ttt = "matlab -nodisplay -nosplash -nojvm -wait -r \"test_m('" + temp_path + "/../aaa');exit\"";
	//system(ttt.c_str());
	//return 0;

	cout << "argc: " << argc << endl;
	cout << "stop at: " << stop_at << endl;

	for (int i = 0; i < img_names.size(); ++i) {
		string test_name;		

		//Big CSV
		//cout << "big" << endl;
		WriterTypeCSV::Pointer csv_big_writer = WriterTypeCSV::New();

		string big_csv_fn = temp_path + "/big_csv.csv";
		csv_big_writer->SetFileName(big_csv_fn);
		csv_big_writer->ColumnHeadersPushBack("\"Dummy\"");
		for (int j = 0; j < data->GetColumnHeaders().size(); ++j) {
			string col_header = "\"" + data->GetColumnHeaders()[j] + "\"";
			csv_big_writer->ColumnHeadersPushBack(col_header);
		}

		for (int j = 1; j < data->GetRowHeaders().size(); ++j) {
			string row_header = "\"" + data->GetRowHeaders()[j] + "\"";
			csv_big_writer->RowHeadersPushBack(row_header);
			//cout << row_header << endl;
		}
		csv_big_writer->SetFieldDelimiterCharacter(';');

		vnl_matrix<double> big_mat(data->GetMatrix().rows()-1, data->GetMatrix().cols());
		for (int j = 1; j < data->GetMatrix().rows(); ++j) {
			for (int k = 0; k < data->GetMatrix().cols(); ++k) {
				big_mat(j - 1, k) = data->GetMatrix()(j, k);
			}
		}
		csv_big_writer->SetInput(&big_mat);
		csv_big_writer->Write();
		//cout << big_mat << endl;
		

		//test csv
		//cout << "test" << endl;
		WriterTypeCSV::Pointer csv_test_writer = WriterTypeCSV::New();

		string test_csv_fn = temp_path + "/test_csv.csv";
		csv_test_writer->SetFileName(test_csv_fn);
		csv_test_writer->ColumnHeadersPushBack("\"Dummy\"");
		for (int j = 0; j < data->GetColumnHeaders().size(); ++j) {
			string col_header = "\"" + data->GetColumnHeaders()[j] + "\"";
			csv_test_writer->ColumnHeadersPushBack(col_header);
		}

		{
			string row_header = "\"" + data->GetRowHeaders()[0] + "\"";
			csv_test_writer->RowHeadersPushBack(row_header);
			test_name = data->GetRowHeaders()[0];
			//cout << row_header << endl;
		}
		csv_test_writer->SetFieldDelimiterCharacter(';');

		vnl_matrix<double> test_mat(1, data->GetMatrix().cols());

		for (int k = 0; k < data->GetMatrix().cols(); ++k) {
			test_mat(0, k) = data->GetMatrix()(0, k);
		}

		csv_test_writer->SetInput(&test_mat);
		csv_test_writer->Write();
		//cout << test_mat << endl;

		// DO STUFF

		cout << "Image left out: " << test_name << " " << i + 1 << "/" << img_names.size() << endl;

		
		//EXTRACT
		string pca_prop_file = big_csv_fn;
		string pca_input_dir = input_img_dir_path;
		string pca_out_dir = temp_path;
		string pca_model_file = output_path + "/model_" + test_name + ".txt";
		string pca_command = "\"\"" + pca_exe + "\"\"" + sp + pca_prop_file + sp + pca_input_dir + sp + pca_out_dir + sp + pca_model_file + sp + "30";
		//cout << pca_command << endl;
		cout << "PCA..." << endl;
		system(pca_command.c_str());

		//PREDICT
		string match_prop_file = test_csv_fn;
		string match_input_dir = input_img_dir_path;
		string match_pca_dir = pca_out_dir;
		string match_output_dir = output_path;
		string match_proj_file = output_path + "/proj_" + test_name + ".txt";
		string match_command = "\"\"" + match_exe + "\"\"" + sp + match_prop_file + sp + match_input_dir + sp + match_pca_dir + sp + match_output_dir + sp + match_proj_file;
		//cout << match_command << endl;
		cout << "predicting..." << endl;
		system(match_command.c_str());

		//SHIFT		
		WriterTypeCSV::Pointer csv_writer = WriterTypeCSV::New();

		string pause_csv_fn = temp_path + "/pause_csv.csv";
		csv_writer->SetFileName(pause_csv_fn);
		csv_writer->ColumnHeadersPushBack("\"Dummy\"");
		for (int j = 0; j < data->GetColumnHeaders().size(); ++j) {
			string col_header = "\"" + data->GetColumnHeaders()[j] + "\"";
			csv_writer->ColumnHeadersPushBack(col_header);
		}

		for (int j = 1; j < data->GetRowHeaders().size(); ++j) {
			string row_header = "\"" + data->GetRowHeaders()[j] + "\"";
			csv_writer->RowHeadersPushBack(row_header);
		}
		string row_header = "\"" + data->GetRowHeaders()[0] + "\"";
		//cout << row_header << endl;
		csv_writer->RowHeadersPushBack(row_header);
		csv_writer->SetFieldDelimiterCharacter(';');

		vnl_matrix<double> pause_mat(data->GetMatrix().rows(), data->GetMatrix().cols());
		for (int j = 1; j < data->GetMatrix().rows(); ++j) {
			for (int k = 0; k < data->GetMatrix().cols(); ++k) {
				pause_mat(j - 1, k) = data->GetMatrix()(j, k);
			}
		}		

		for (int k = 0; k < data->GetMatrix().cols(); ++k) {
			pause_mat(data->GetMatrix().rows() - 1, k) = data->GetMatrix()(0, k);
		}

		csv_writer->SetInput(&pause_mat);
		csv_writer->Write();

		csv_reader = ReaderTypeCSV::New();
		csv_reader->SetFileName(pause_csv_fn);
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
		data = csv_reader->GetOutput(); 

		
		if (stop_at.compare(test_name) == 0) break;

	}


	cout << "ALL DONE!" << endl;
	return EXIT_SUCCESS;
	
}

