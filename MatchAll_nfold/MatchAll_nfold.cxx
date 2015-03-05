

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

	if (argc < 8) {
		cout << "Please specify:" << endl;
		cout << " k" << endl;
		cout << " CSV File" << endl;
		cout << " Input folder" << endl;
		cout << " Temp Folder" << endl;
		cout << " Output Path" << endl;
		cout << " PCA.exe" << endl;
		//cout << " sigpix.m" << endl;
		cout << " MatchModel.exe" << endl;
		//cout << " [STOP AT]" << endl;
		
		//cout << " [Reconstruct]" << endl;
		return EXIT_FAILURE;
	}

	double pval_thresh = 0.01;

	//string asdf = "start ../Predict.exe";
	//system(asdf.c_str());
	//return 0;

	typedef CSVNumericObjectFileWriter<double> WriterTypeCSV;
	typedef itk::CSVArray2DFileReader<double> ReaderTypeCSV;

	int k = atoi(argv[1]);
	string csv_path(argv[2]);
	string input_img_dir_path(argv[3]);
	string temp_path(argv[4]);
	string output_path(argv[5]);
	string pca_exe(argv[6]);
	//string sigpix_m(argv[6]);
	//sigpix_m = sigpix_m.substr(0, sigpix_m.size() - 2);
	string match_exe(argv[7]);
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

	//string stop_at;
	//if (argc >= 8) {
	//	stop_at = argv[7];		
	//} else {
	//	stop_at = img_names.back();
	//}

	//string ttt = "matlab -nodisplay -nosplash -nojvm -wait -r \"test_m('" + temp_path + "/../aaa');exit\"";
	//system(ttt.c_str());
	//return 0;

	cout << "argc: " << argc << endl;
	//cout << "stop at: " << stop_at << endl;

	int n_images = data->GetRowHeaders().size();
	cout << "n_images: " << n_images << endl;
	double partition_size = double(n_images) / k;
	int partition_size_int = partition_size;
	int out_begin = 0;
	int out_end = partition_size_int;

	for (int i = 0; i < k; ++i) {
		if (i == k - 1) {
			out_end = n_images;
		}

		cout << "begin: " << out_begin << endl;
		cout << "end: " << out_end << endl;


		//test csv
		WriterTypeCSV::Pointer csv_test_writer = WriterTypeCSV::New();

		string test_csv_fn = temp_path + "/test_csv.csv";
		csv_test_writer->SetFileName(test_csv_fn);
		csv_test_writer->ColumnHeadersPushBack("\"Dummy\"");
		for (int j = 0; j < data->GetColumnHeaders().size(); ++j) {
			string col_header = "\"" + data->GetColumnHeaders()[j] + "\"";
			csv_test_writer->ColumnHeadersPushBack(col_header);
		}

		{
			for (int j = out_begin; j < out_end; ++j) {
				string row_header = "\"" + data->GetRowHeaders()[j] + "\"";
				csv_test_writer->RowHeadersPushBack(row_header);
			}
		}
		csv_test_writer->SetFieldDelimiterCharacter(';');

		vnl_matrix<double> test_mat(out_end - out_begin, data->GetMatrix().cols());

	
		for (int j = out_begin; j < out_end; ++j) {
			for (int m = 0; m < data->GetMatrix().cols(); ++m) {
				test_mat(j - out_begin, m) = data->GetMatrix()(j, m);
			}
		}

		csv_test_writer->SetInput(&test_mat);
		csv_test_writer->Write();


	

		//train CSV
		//cout << "big" << endl;
		WriterTypeCSV::Pointer csv_train_writer = WriterTypeCSV::New();

		string big_csv_fn = temp_path + "/train_csv.csv";
		csv_train_writer->SetFileName(big_csv_fn);
		csv_train_writer->ColumnHeadersPushBack("\"Dummy\"");
		for (int j = 0; j < data->GetColumnHeaders().size(); ++j) {
			string col_header = "\"" + data->GetColumnHeaders()[j] + "\"";
			csv_train_writer->ColumnHeadersPushBack(col_header);
		}

		for (int j = 0; j < data->GetRowHeaders().size(); ++j) {
			if (j >= out_begin && j < out_end) continue;
			string row_header = "\"" + data->GetRowHeaders()[j] + "\"";
			csv_train_writer->RowHeadersPushBack(row_header);
			//cout << row_header << endl;
		}
		csv_train_writer->SetFieldDelimiterCharacter(';');

		vnl_matrix<double> big_mat(data->GetMatrix().rows()-1, data->GetMatrix().cols());
		int j_idx = 0;
		for (int j = 0; j < data->GetMatrix().rows(); ++j) {
			if (j >= out_begin && j < out_end) continue;
			for (int k = 0; k < data->GetMatrix().cols(); ++k) {
				big_mat(j_idx, k) = data->GetMatrix()(j, k);
			}
			++j_idx;
		}
		csv_train_writer->SetInput(&big_mat);
		csv_train_writer->Write();
		//cout << big_mat << endl;
		

		
		//cout << test_mat << endl;

		// DO STUFF

		cout << "Images left out: " << endl;
		for (int j = out_begin; j < out_end; ++j) {
			cout << "   " << data->GetRowHeaders()[j] << endl;
		}
		string test_name = data->GetRowHeaders()[out_begin];


		
		//EXTRACT
		string pca_prop_file = big_csv_fn;
		string pca_input_dir = input_img_dir_path;
		string pca_out_dir = temp_path;
		string pca_model_file = output_path + "/model_" + test_name + ".txt";
		string pca_command = "\"\"" + pca_exe + "\"\"" + sp + pca_prop_file + sp + pca_input_dir + sp + pca_out_dir + sp + pca_model_file + sp + "30";
		cout << pca_command << endl;
		//cout << "PCA..." << endl;
		//system(pca_command.c_str());

		//PREDICT
		string match_prop_file = test_csv_fn;
		string match_input_dir = input_img_dir_path;
		string match_pca_dir = pca_out_dir;
		string match_output_dir = output_path;
		string match_proj_file = output_path + "/proj_" + test_name + ".txt";
		string match_command = "\"\"" + match_exe + "\"\"" + sp + match_prop_file + sp + match_input_dir + sp + match_pca_dir + sp + match_output_dir + sp + match_proj_file;
		cout << match_command << endl;
		//cout << "predicting..." << endl;
		//system(match_command.c_str());

		
		out_begin += partition_size_int;
		out_end += partition_size_int;

	}


	cout << "ALL DONE!" << endl;
	return EXIT_SUCCESS;
	
}

