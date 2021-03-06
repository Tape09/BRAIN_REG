
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMaskImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkDirectory.h"
#include "itkFileTools.h"
#include "itkExceptionObject.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkTransformFileReader.h"
#include "itkTransformFactoryBase.h"
#include "itkBSplineTransform.h"
#include "itkImagePCAShapeModelEstimator.h"
#include "itkIterativeInverseDisplacementFieldImageFilter.h"
#include "itkDisplacementFieldTransform.h"
#include "itkScalarImageKmeansImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkImagePCADecompositionCalculator.h"
#include "itkCSVArray2DFileReader.h"
#include "itkMultiplyImageFilter.h"
#include "itkAddImageFilter.h"

#include <fstream>
#include <iostream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>



typedef itk::Image< double, 3 > ImageType;
typedef itk::Image< unsigned char, 3 > ImageTypeUC;
typedef itk::Image< double, 2 > ImageTypePCA;

typedef itk::AddImageFilter<ImageTypePCA> AddFilterType;
typedef itk::MultiplyImageFilter<ImageTypePCA> MultFilterType;

typedef unsigned uint;

const int k = 20;

const unsigned int spline_order = 3;
typedef itk::BSplineTransform<double, 3, spline_order> TransformTypeBSpline;

typedef itk::Vector<double, 3> VectorType;
typedef itk::Point<double, 3> PointType;
//typedef itk::Image< PointType, 2 > ImageTypePCA;
typedef itk::Image< VectorType, 3 > DisplacementFieldType;
typedef itk::DisplacementFieldTransform<double, 3> TransformTypeDis;

//typedef itk::ImageFileWriter< ImageTypePCAtest > WriterTypePCAtest;

typedef ImageType::IndexType ImageIndexType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileReader<ImageTypePCA> ReaderTypePCA;
typedef itk::ImageFileReader<ImageTypeUC> ReaderTypeUC;
typedef itk::ImageFileWriter< ImageType  > WriterType;
typedef itk::ImageFileWriter< ImageTypePCA > WriterTypePCA;
typedef itk::ImageFileWriter< ImageTypeUC > WriterTypeUC;
typedef itk::ImageDuplicator< ImageType > DuplicatorType;
typedef itk::ImageDuplicator< ImageTypePCA > DuplicatorTypePCA;
typedef itk::ImagePCADecompositionCalculator<ImageTypePCA, ImageTypePCA> ProjectorType;

using namespace std;
using namespace itk;


void copy_img(ImageType::Pointer & from_img, ImageType::Pointer & to_img);
int count_brain_pix(ImageType::Pointer & img);
int diff_img(ImageType::Pointer & from_img, ImageType::Pointer & to_img, ImageType::Pointer & out_img);
int getCommonPix(ImageType::Pointer & in_img, ImageTypeUC::Pointer & common_map);
void getInverseTfm(ImageType::Pointer & fixed_image, TransformTypeBSpline::Pointer & tfm, TransformTypeDis::Pointer & itfm);
void getLabelImage(ImageType::Pointer & image, Image<unsigned char, 3>::Pointer & out_image, int n_classes);
void apply_mask(ImageTypeUC::Pointer & in_img, ImageTypeUC::Pointer & common_map);

int main(int argc, char *argv[]) {

	itk::TimeProbe totalTimeClock;
	totalTimeClock.Start();

	if (argc < 5) {
		cout << "Please specify:" << endl;
		cout << " CSV File" << endl;
		cout << " Input folder" << endl;
		cout << " Input PCA folder" << endl;
		cout << " Output folder" << endl;
		cout << " [Reconstruct]" << endl;
		return EXIT_FAILURE;
	}
	DuplicatorType::Pointer duplicator = DuplicatorType::New();
	DuplicatorTypePCA::Pointer duplicator_pca = DuplicatorTypePCA::New();


	typedef itk::CSVArray2DFileReader<double> ReaderTypeCSV;
	ReaderTypeCSV::Pointer csv_reader = ReaderTypeCSV::New();

	csv_reader->SetFileName(argv[1]);
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
		//cout << prop_names[i] << endl;
	}

	vector<string> img_names;
	for (int i = 0; i < data->GetRowHeaders().size(); ++i) {
		img_names.push_back(data->GetRowHeaders()[i]);
	}

	string input_img_dir(argv[2]);
	string input_pca_path(argv[3]);
	string output_dir_path(argv[4]);
	bool reconstruct_images = false;
	if (argc == 6) reconstruct_images = true;

	Directory::Pointer output_dir = Directory::New();

	try {
		output_dir->Load(output_dir_path.c_str());
	} catch (ExceptionObject & err) {
		cerr << err << endl;
		return EXIT_FAILURE;
	}

	if (output_dir->GetNumberOfFiles() == 0) {
		FileTools::CreateDirectoryA(output_dir_path.c_str());
	}

	// Set up projector
	ProjectorType::Pointer projector = ProjectorType::New();
	ProjectorType::BasisImagePointerVector basis_image_vector;


	// READ MASK
	ReaderTypeUC::Pointer mask_reader = ReaderTypeUC::New();
	mask_reader->SetFileName(input_pca_path + "/common_mask.nii");

	try {
		mask_reader->Update();
	} catch (ExceptionObject & e) {
		cerr << e << endl;
		cerr << "Cant read mask" << endl;
		return EXIT_FAILURE;
	}

	ImageTypeUC::Pointer common_map = ImageTypeUC::New();
	common_map = mask_reader->GetOutput();

	// READ MEAN
	ReaderTypePCA::Pointer mean_reader = ReaderTypePCA::New();
	mean_reader->SetFileName(input_pca_path + "/mean.mhd");

	try {
		mean_reader->Update();
	} catch (ExceptionObject & e) {
		cerr << e << endl;
		cerr << "Cant read mean" << endl;
		return EXIT_FAILURE;
	}

	ImageTypePCA::Pointer mean_image = ImageTypePCA::New();
	mean_image = mean_reader->GetOutput();

	int n_common = mean_image->GetBufferedRegion().GetSize()[1];
	cout << n_common << endl;

	// READ PCs

	ReaderTypePCA::Pointer pc_reader = ReaderTypePCA::New();

	int counter = 1;
	while (true) {
		string fn = input_pca_path + "/PC_" + to_string(counter) + ".mhd";
		
		pc_reader->SetFileName(fn);

		try {
			pc_reader->Update();
		} catch (ExceptionObject & ) {
			break;
		}
		cout << fn << endl;

		duplicator_pca->SetInputImage(pc_reader->GetOutput());
		duplicator_pca->Update();
		basis_image_vector.push_back(duplicator_pca->GetOutput());
		++counter;
	}

	ofstream myfile;
	string out_fn = output_dir_path + "/results.txt";
	myfile.precision(25);
	myfile.open(out_fn, ios::trunc);

	ofstream resfile;
	string res_fn = output_dir_path + "/res_matlab.txt";
	resfile.precision(25);
	resfile.open(res_fn, ios::trunc);
	for (int i = 0; i < img_names.size(); ++i) {


		// READ IMAGE
		string img_name = fn_prefix + img_names[i] + fn_img_suffix;
		string img_path = input_img_dir + "/images/" + img_name;

		cout << "Processing Image " << img_name << "- " << i + 1 << "/" << img_names.size() << endl;
		myfile << img_name << endl;

		ReaderType::Pointer reader = ReaderType::New();
		reader->SetFileName(img_path);

		ImageType::Pointer input_image = ImageType::New();
		try {
			reader->Update();
		} catch (ExceptionObject & e) {
			cerr << e << endl;
			cerr << "Cant read input image" << endl;
			return EXIT_FAILURE;
		}
		input_image = reader->GetOutput();
		duplicator->SetInputImage(input_image);
		duplicator->Update();
		ImageType::Pointer input_image_copy = ImageType::New();
		input_image_copy = duplicator->GetOutput();

		// READ TFM
		string tfm_name = fn_prefix + img_names[i] + fn_tfm_suffix;
		string tfm_path = input_img_dir + "/transforms/" + tfm_name;

		itk::TransformFileReader::Pointer tfm_reader = itk::TransformFileReader::New();
		tfm_reader->SetFileName(tfm_path);

		try {
			tfm_reader->Update();
		} catch (ExceptionObject & e) {
			cerr << e << endl;
			cerr << "Cant read input transform" << endl;
			return EXIT_FAILURE;
		}

		TransformTypeBSpline::Pointer tfm = static_cast<TransformTypeBSpline*>(tfm_reader->GetTransformList()->back().GetPointer());
		//TransformTypeDis::Pointer itfm = TransformTypeDis::New();
		//getInverseTfm(input_image_copy, tfm, itfm);

		// Create input image	
		ImageTypePCA::Pointer input_PCA_image = ImageTypePCA::New();
		ImageTypePCA::RegionType region;
		itk::Size<2> sz;
		sz[0] = 3;
		sz[1] = n_common;
		region.SetSize(sz);

		input_PCA_image->SetRegions(region);
		input_PCA_image->Allocate();

		itk::ImageRegionIteratorWithIndex<ImageTypeUC> common_itr(common_map, common_map->GetBufferedRegion());
		itk::ImageRegionIteratorWithIndex<ImageTypePCA> pcaimg_itr(input_PCA_image, input_PCA_image->GetBufferedRegion());

		double common_val;
		ImageTypeUC::IndexType common_idx;
		Point<double, 3> img_point;
		Point<double, 3> temp_point;
		while (!common_itr.IsAtEnd()) {
			common_val = common_itr.Get();
			if (common_val > 0.5) {
				common_idx = common_itr.GetIndex();
				common_map->TransformIndexToPhysicalPoint(common_idx, temp_point);
				//img_point = itfm->TransformPoint(img_point);
				img_point = tfm->TransformPoint(temp_point);
				pcaimg_itr.Set(img_point[0]);
				++pcaimg_itr;
				pcaimg_itr.Set(img_point[1]);
				++pcaimg_itr;
				pcaimg_itr.Set(img_point[2]);
				++pcaimg_itr;
			}
			++common_itr;
		}


		// PLUG IT ALL IN
		projector->SetImage(input_PCA_image);
		projector->SetMeanImage(mean_image);
		projector->SetBasisImages(basis_image_vector);

		projector->Compute();

		ProjectorType::BasisVectorType projection = projector->GetProjection();
		//cout << projection.size() << endl;

		// PROPERTIES


		Eigen::Matrix<double, Eigen::Dynamic, 1> p;
		p.resize(projection.size(), 1);
		for (int j = 0; j < projection.size(); ++j) {
			p(j, 0) = projection[j];
		}

		Eigen::Matrix<double, 1, Eigen::Dynamic> w;
		w.resize(1, projection.size());

		Eigen::Matrix<double, 1, 1> r;

		
		
		vector<double> temp_prop;
		string temp;
		string line;
		ifstream infile(input_pca_path + "/properties.txt");
		if (infile.is_open()) {
			int p_idx = 0;
			while (getline(infile, line)) {
				istringstream ss(line);

				if (!getline(ss, temp, ' ')) continue;

				cout << temp << ": ";
				myfile << temp << " ";

				temp_prop.clear();
				while (ss) {
					if (!getline(ss, temp, ' ')) break;
					temp_prop.push_back(atof(temp.c_str()));
				}

				if (temp_prop.size() != projection.size()) {
					cerr << "ERROR: projection size does not match properties size: " << temp_prop.size() << " " << projection.size() << endl;
					return EXIT_FAILURE;
				}

				for (int j = 0; j < temp_prop.size(); ++j) {
					w(0, j) = temp_prop[j];
				}

				//cout << endl;
				//cout << "w: " << endl << w << endl;
				//cout << "p: " << endl << p << endl;
				r = w*p;
				resfile << data->GetData(img_names[i], prop_names[p_idx]) << " " << r << endl;


				cout << r << endl;
				myfile << r << endl;
				++p_idx;

			}
			infile.close();
		}

		//if (temp_prop.size() != sz) {
		//	cerr << "bad input file? expected " << sz << " properties" << endl;
		//	exit(EXIT_FAILURE);
		//}

		//for (int i = 0; i < temp_prop.size(); ++i) {
		//	b(i) = temp_prop[i];
		//}


		myfile << "projection ";
		for (int i = 0; i < projection.size(); ++i) {
			myfile << projection[i] << " ";
		}
		myfile << endl << endl;
		

		if (reconstruct_images) {
			// GET RECONSTRUCTED IMAGE
			cout << "Reconstructing image..." << endl;			

			ImageTypePCA::Pointer re_PCA_image = ImageTypePCA::New();

			duplicator_pca->SetInputImage(projector->GetMeanImage());
			duplicator_pca->Update();
			re_PCA_image = duplicator_pca->GetOutput();

			AddFilterType::Pointer addFilter = AddFilterType::New();
			MultFilterType::Pointer multFilter = MultFilterType::New();

			cout << "Creating image from model..." << endl;
			
			for (int j = 0; j < projection.size(); ++j) { // iterate over basis_image_vector and projection, use AddFilterType, MultFilterType
				
				multFilter->SetInput(projector->GetBasisImages()[j]);
				multFilter->SetConstant(projection[j]);
				multFilter->Update();

				addFilter->SetInput1(re_PCA_image);
				addFilter->SetInput2(multFilter->GetOutput());
				addFilter->Update();

				re_PCA_image = addFilter->GetOutput();
				cout << "Progress: " << ((j + 1) * 100 / projection.size()) << "%\r";

			}

			// Decode
			cout << "Changing formats and coloring..." << endl;

			ImageType::Pointer re_image = ImageType::New();
			re_image->SetRegions(input_image->GetBufferedRegion());
			re_image->SetDirection(input_image->GetDirection());
			re_image->SetOrigin(input_image->GetOrigin());
			re_image->SetSpacing(input_image->GetSpacing());
			re_image->Allocate();
			re_image->FillBuffer(0);

			itk::ImageRegionIteratorWithIndex<ImageTypeUC> c_itr(common_map, common_map->GetBufferedRegion());
			itk::ImageRegionIteratorWithIndex<ImageType> img_itr(input_image, input_image->GetBufferedRegion());
			itk::ImageRegionIteratorWithIndex<ImageTypePCA> repcaimg_itr(re_PCA_image, re_PCA_image->GetBufferedRegion());

			ImageType::IndexType index;
			Point<double, 3> point;

			int count = 1;
			while (!c_itr.IsAtEnd()) {
				common_val = c_itr.Get();

				if (common_val > 0.5) {
					double value = img_itr.Get();
					
					point[0] = repcaimg_itr.Get();
					++repcaimg_itr;
					point[1] = repcaimg_itr.Get();
					++repcaimg_itr;
					point[2] = repcaimg_itr.Get();
					++repcaimg_itr;				

					re_image->TransformPhysicalPointToIndex(point, index);
					//cout << index << endl;
					//re_image->SetPixel(index, value);
					bool setpix = true;
					for (int j = 0; j < 3; ++j) {
						if (index[j] < 0) {
							//index[j] = 0;
							setpix = false;
						}

						if (index[j] >= input_image->GetBufferedRegion().GetSize()[j]) {
							//index[j] = input_image->GetBufferedRegion().GetSize()[j] - 1;
							setpix = false;
						}
					}


					try {
						//re_image->SetPixel(index, 1);
						if (setpix)	re_image->SetPixel(index, value);
					} catch (itk::ExceptionObject & e) {
						cerr << e << endl;
						exit(EXIT_FAILURE);
					}
					
					if (count % (n_common / 100) == 0) {
						cout << "Progress: " << count * 100 / n_common << "%\r";
					}

					++count;
				}

				++c_itr;
				++img_itr;
			}

			WriterType::Pointer writer = WriterType::New();
			writer->SetInput(re_image);
			string re_fn = output_dir_path + "/re_" + img_name;
			writer->SetFileName(re_fn);
			writer->Update();
		}


	}
	myfile.close();
	resfile.close();
	cout << "ALL DONE!" << endl;
	return EXIT_SUCCESS;
	// ________________________________

	//unsigned int n_pcomponents;
	//try {
	//	n_pcomponents = stoi(argv[1]);
	//} catch (exception & e) {
	//	cerr << e.what() << endl;
	//	cerr << "N Principal Components invalid" << endl;
	//	return EXIT_FAILURE;
	//}


	//string output_dir_path(argv[2]);

	//vector<string> input_dir_paths;
	//for (int i = 3; i < argc; ++i) {
	//	input_dir_paths.push_back(argv[i]);
	//}

	//vector<ImageType::Pointer> images;
	//vector<TransformTypeDis::Pointer> inv_tfms;


	//Directory::Pointer input_img_dir = Directory::New();
	//Directory::Pointer input_tfm_dir = Directory::New();
	//Directory::Pointer output_dir = Directory::New();

	//try {
	//	output_dir->Load(output_dir_path.c_str());
	//} catch (ExceptionObject & err) {
	//	cerr << err << endl;
	//	return EXIT_FAILURE;
	//}

	//if (output_dir->GetNumberOfFiles() == 0) {
	//	FileTools::CreateDirectoryA(output_dir_path.c_str());
	//}

	//itk::TransformFileReader::Pointer tfm_reader = itk::TransformFileReader::New();
	//ReaderType::Pointer img_reader = ReaderType::New();
	//WriterType::Pointer img_writer = WriterType::New();
	//DuplicatorType::Pointer duplicator = DuplicatorType::New();

	//for (int d = 0; d < input_dir_paths.size(); ++d) {
	//	cout << "Processing directory " << input_dir_paths[d] << "- " << d + 1 << "/" << input_dir_paths.size() << endl;


	//	// DIRECTORY STUFF
	//	string input_img_dir_path(input_dir_paths[d] + "/images");
	//	string input_tfm_dir_path(input_dir_paths[d] + "/transforms");

	//	try {
	//		input_img_dir->Load(input_img_dir_path.c_str());
	//		input_tfm_dir->Load(input_tfm_dir_path.c_str());
	//	} catch (ExceptionObject & err) {
	//		cerr << err << endl;
	//		return EXIT_FAILURE;
	//	}

	//	if (input_img_dir->GetNumberOfFiles() == 0) {
	//		cerr << "Input Directory Invalid" << endl;
	//		return EXIT_FAILURE;
	//	}

	//	if (input_tfm_dir->GetNumberOfFiles() == 0) {
	//		cerr << "Input Directory Invalid" << endl;
	//		return EXIT_FAILURE;
	//	}
	//	// 

	//	string img_path;
	//	string tfm_path;
	//	string img_name;
	//	string tfm_name;
	//	string output_img_path;
	//	string output_tfm_path;
	//	Directory::Pointer test_file = Directory::New();
	//	for (int i = 2; i < input_img_dir->GetNumberOfFiles(); ++i) {

	//		img_name = input_img_dir->GetFile(i);
	//		cout << "Processing Image " << img_name << "- " << i - 1 << "/" << input_img_dir->GetNumberOfFiles() - 2 << endl;

	//		img_path = input_img_dir_path + "/" + img_name;
	//		test_file->Load(img_path.c_str());


	//		if (test_file->GetNumberOfFiles() > 0) continue;

	//		int pos;
	//		pos = img_name.find_first_of(".");
	//		tfm_name = img_name.substr(0, pos) + ".tfm";
	//		tfm_path = input_tfm_dir_path + "/" + tfm_name;


	//		try {
	//			img_reader->SetFileName(img_path);
	//			img_reader->Update();
	//			duplicator->SetInputImage(img_reader->GetOutput());
	//			duplicator->Update();
	//			images.push_back(duplicator->GetOutput());

	//			tfm_reader->SetFileName(tfm_path);
	//			tfm_reader->Update();

	//			TransformTypeBSpline::Pointer tfm = static_cast<TransformTypeBSpline*>(tfm_reader->GetTransformList()->back().GetPointer());
	//			TransformTypeDis::Pointer itfm = TransformTypeDis::New();

	//			getInverseTfm(images.back(), tfm, itfm);

	//			inv_tfms.push_back(itfm);
	//		} catch (ExceptionObject & err) {
	//			cerr << err << endl;
	//			return EXIT_FAILURE;
	//		}
	//	}
	//}

	//if (images.size() < 2) {
	//	cerr << "NOT ENOUGH IMAGES" << endl;
	//	return EXIT_FAILURE;
	//}

	////duplicator->SetInputImage(images[0]);
	////duplicator->Update();

	//ImageTypeUC::Pointer common_map = ImageTypeUC::New();

	//getLabelImage(images[0], common_map, k);

	////ImageType::Pointer common_map = ImageType::New();
	////common_map = duplicator->GetOutput();	

	//cout << "Calculating common pixels..." << endl;

	//int n_common;
	//for (int i = 1; i < images.size(); ++i) {
	//	n_common = getCommonPix(images[i], common_map);
	//	cout << "n common pixels: " << n_common << endl;
	//}

	//WriterTypeUC::Pointer mask_writer = WriterTypeUC::New();
	//string mask_fn = output_dir_path + "/" + "common_mask.nii";
	//mask_writer->SetFileName(mask_fn);
	//mask_writer->SetInput(common_map);
	//mask_writer->Update();


	//vector<ImageTypePCA::Pointer> images_PCA(images.size());
	//ImageTypePCA::RegionType region;
	//itk::Size<2> sz;
	//sz[0] = 3;
	//sz[1] = n_common;
	//region.SetSize(sz);

	//for (int i = 0; i < images.size(); ++i) {
	//	images_PCA[i] = ImageTypePCA::New();
	//	images_PCA[i]->SetRegions(region);
	//	images_PCA[i]->Allocate();

	//	itk::ImageRegionIteratorWithIndex<ImageType> img_itr(images[i], images[i]->GetBufferedRegion());
	//	itk::ImageRegionIteratorWithIndex<ImageTypeUC> common_itr(common_map, common_map->GetBufferedRegion());
	//	itk::ImageRegionIteratorWithIndex<ImageTypePCA> pcaimg_itr(images_PCA[i], images_PCA[i]->GetBufferedRegion());

	//	double common_val;
	//	ImageType::IndexType img_idx;
	//	Point<double, 3> img_point;
	//	while (!img_itr.IsAtEnd()) {
	//		common_val = common_itr.Get();
	//		if (common_val > 0) {
	//			img_idx = img_itr.GetIndex();
	//			images[i]->TransformIndexToPhysicalPoint(img_idx, img_point);
	//			img_point = inv_tfms[i]->TransformPoint(img_point);
	//			pcaimg_itr.Set(img_point[0]);
	//			++pcaimg_itr;
	//			pcaimg_itr.Set(img_point[1]);
	//			++pcaimg_itr;
	//			pcaimg_itr.Set(img_point[2]);
	//			++pcaimg_itr;
	//		}
	//		++img_itr;
	//		++common_itr;
	//	}
	//}

	//n_pcomponents = min(uint(n_pcomponents), uint(images_PCA.size()));

	//typedef itk::ImagePCAShapeModelEstimator<ImageTypePCA, ImageTypePCA >  EstimatorType;
	//EstimatorType::Pointer estimator = EstimatorType::New();
	//estimator->SetNumberOfTrainingImages(images_PCA.size());
	//estimator->SetNumberOfPrincipalComponentsRequired(n_pcomponents);

	//for (int i = 0; i < images_PCA.size(); ++i) {
	//	estimator->SetInput(i, images_PCA[i]);
	//}

	//itk::TimeProbe clock;
	//clock.Start();
	//cout << "Preforming PCA analysis with " << n_pcomponents << " components..." << endl;
	//estimator->Update();
	//clock.Stop();
	//cout << "Done!" << endl;
	//cout << "Time taken: " << clock.GetTotal() << "s" << endl;


	//string fn;
	//fn = output_dir_path + "/mean.mhd";
	//WriterTypePCA::Pointer pca_writer = WriterTypePCA::New();
	//pca_writer->SetFileName(fn);
	//pca_writer->SetInput(estimator->GetOutput(0));

	//try {
	//	pca_writer->Update();
	//} catch (ExceptionObject & err) {
	//	cerr << err << endl;
	//	return EXIT_FAILURE;
	//}




	//for (int i = 1; i < n_pcomponents + 1; ++i) {
	//	fn = output_dir_path + "/PC_" + to_string(i) + ".mhd";
	//	pca_writer->SetFileName(fn);
	//	pca_writer->SetInput(estimator->GetOutput(i));
	//	try {
	//		pca_writer->Update();
	//	} catch (ExceptionObject & err) {
	//		cerr << err << endl;
	//		return EXIT_FAILURE;
	//	}
	//}

	//totalTimeClock.Stop();
	//cout << "ALL DONE!" << endl;
	//cout << "Total time taken: " << totalTimeClock.GetTotal() << "s" << endl;

	//return EXIT_SUCCESS;
}

void copy_img(ImageType::Pointer & from_img, ImageType::Pointer & to_img) {
	itk::ImageRegionIteratorWithIndex<ImageType> from_imgIterator(from_img, from_img->GetBufferedRegion());
	itk::ImageRegionIteratorWithIndex<ImageType> to_imgIterator(to_img, to_img->GetBufferedRegion());


	while (!from_imgIterator.IsAtEnd()) {

		double val = from_imgIterator.Get();
		if (val > 0) {
			to_imgIterator.Set(255);
		} else {
			to_imgIterator.Set(0);
		}
		++from_imgIterator;
		++to_imgIterator;
	}

}

int count_brain_pix(ImageType::Pointer & img) {
	itk::ImageRegionIteratorWithIndex<ImageType> imgIterator(img, img->GetBufferedRegion());

	int counter = 0;
	while (!imgIterator.IsAtEnd()) {

		double val = imgIterator.Get();
		if (val > 0.1) {
			++counter;
		}
		++imgIterator;
	}

	return counter;
}

int diff_img(ImageType::Pointer & from_img, ImageType::Pointer & to_img, ImageType::Pointer & out_img) {
	itk::ImageRegionIteratorWithIndex<ImageType> from_imgIterator(from_img, from_img->GetBufferedRegion());
	itk::ImageRegionIteratorWithIndex<ImageType> to_imgIterator(to_img, to_img->GetBufferedRegion());
	itk::ImageRegionIteratorWithIndex<ImageType> out_imgIterator(out_img, out_img->GetBufferedRegion());

	int counter = 0;

	while (!from_imgIterator.IsAtEnd()) {

		double val_from = from_imgIterator.Get();
		double val_to = to_imgIterator.Get();
		if (val_from > 0.1 != val_to > 0.1) {
			out_imgIterator.Set(val_to);
			++counter;
		} else {
			out_imgIterator.Set(0);
		}
		++from_imgIterator;
		++to_imgIterator;
		++out_imgIterator;
	}
	return counter;

}

int getCommonPix(ImageType::Pointer & in_img, ImageTypeUC::Pointer & common_map) {

	ImageTypeUC::Pointer label_image = ImageTypeUC::New();
	getLabelImage(in_img, label_image, k);

	itk::ImageRegionIteratorWithIndex<ImageTypeUC> in_imgIterator(label_image, label_image->GetBufferedRegion());
	itk::ImageRegionIteratorWithIndex<ImageTypeUC> common_imgIterator(common_map, common_map->GetBufferedRegion());

	int counter = 0;

	while (!in_imgIterator.IsAtEnd()) {

		double val_in = in_imgIterator.Get();
		double val_common = common_imgIterator.Get();
		if (val_in > 0.1 && val_common > 0.1) {
			common_imgIterator.Set(1);
			++counter;
		} else {
			common_imgIterator.Set(0);
		}

		++in_imgIterator;
		++common_imgIterator;
	}

	return counter;

}

void getInverseTfm(ImageType::Pointer & fixed_image, TransformTypeBSpline::Pointer & tfm, TransformTypeDis::Pointer & itfm) {
	itk::TimeProbe clock;
	clock.Start();
	// inverse transform

	cout << "Calculating displacement field..." << endl;
	DisplacementFieldType::Pointer field = DisplacementFieldType::New();
	field->SetRegions(fixed_image->GetBufferedRegion());
	field->SetOrigin(fixed_image->GetOrigin());
	field->SetSpacing(fixed_image->GetSpacing());
	field->SetDirection(fixed_image->GetDirection());
	field->Allocate();

	typedef itk::ImageRegionIterator< DisplacementFieldType > FieldIterator;
	FieldIterator fi(field, fixed_image->GetBufferedRegion());

	fi.GoToBegin();

	TransformTypeBSpline::InputPointType fixedPoint;
	TransformTypeBSpline::OutputPointType movingPoint;
	DisplacementFieldType::IndexType index;

	VectorType displacement;

	while (!fi.IsAtEnd()) {
		index = fi.GetIndex();
		field->TransformIndexToPhysicalPoint(index, fixedPoint);
		movingPoint = tfm->TransformPoint(fixedPoint);
		displacement = movingPoint - fixedPoint;
		fi.Set(displacement);
		++fi;
	}

	cout << "Displacement field calculated!" << endl;
	cout << "Calculating inverse displacement field..." << endl;

	typedef IterativeInverseDisplacementFieldImageFilter<DisplacementFieldType, DisplacementFieldType> IDFImageFilter;
	IDFImageFilter::Pointer IDF_filter = IDFImageFilter::New();
	IDF_filter->SetInput(field);
	IDF_filter->SetStopValue(1);
	IDF_filter->SetNumberOfIterations(50);

	try {
		IDF_filter->Update();
	} catch (ExceptionObject & err) {
		cerr << err << endl;
		exit(EXIT_FAILURE);
	}
	clock.Stop();

	cout << "Inverse displacement field calculated!" << endl;
	std::cout << "Time Taken: " << clock.GetTotal() << "s" << std::endl;

	itfm->SetDisplacementField(IDF_filter->GetOutput());
}

void getLabelImage(ImageType::Pointer & image, Image<unsigned char, 3>::Pointer & out_image, int n_classes) {

	typedef itk::ScalarImageKmeansImageFilter< ImageType > KMeansFilterType;
	KMeansFilterType::Pointer kmeansFilter = KMeansFilterType::New();
	kmeansFilter->SetInput(image);

	const unsigned int useNonContiguousLabels = 0;
	kmeansFilter->SetUseNonContiguousLabels(useNonContiguousLabels);

	typedef MinimumMaximumImageCalculator<ImageType> imageCalculatorType;
	imageCalculatorType::Pointer calculator = imageCalculatorType::New();
	calculator->SetImage(image);
	calculator->Compute();
	const double minIntensity = calculator->GetMinimum();
	const double maxIntensity = calculator->GetMaximum();
	KMeansFilterType::ParametersType initialMeans(n_classes);
	for (int i = 0; i < n_classes; ++i) {
		initialMeans[i] = minIntensity + double(i * (maxIntensity - minIntensity)) / (n_classes - 1);
		//cout << initialMeans[i] << endl;
	}

	for (unsigned int i = 0; i < n_classes; ++i) {
		kmeansFilter->AddClassWithInitialMean(initialMeans[i]);
	}

	kmeansFilter->Update();

	out_image = kmeansFilter->GetOutput();
	//KMeansFilterType::ParametersType eMeans = kmeansFilter->GetFinalMeans();

	//for (unsigned int i = 0; i < eMeans.Size(); ++i) {
	//	std::cout << "cluster[" << i << "] ";
	//	std::cout << " estimated mean : " << eMeans[i] << std::endl;
	//}

}

void apply_mask(ImageType::Pointer & in_img, ImageTypeUC::Pointer & common_map) {
	ImageTypeUC::Pointer label_image = ImageTypeUC::New();
	getLabelImage(in_img, label_image, k);

	itk::ImageRegionIteratorWithIndex<ImageTypeUC> in_imgIterator(label_image, label_image->GetBufferedRegion());
	itk::ImageRegionIteratorWithIndex<ImageTypeUC> common_imgIterator(common_map, common_map->GetBufferedRegion());

	int counter = 0;

	while (!in_imgIterator.IsAtEnd()) {

		double val_in = in_imgIterator.Get();
		double val_common = common_imgIterator.Get();
		if (val_common < 0.5) {
			in_imgIterator.Set(1);
			++counter;
		} else {
			common_imgIterator.Set(0);
		}

		++in_imgIterator;
		++common_imgIterator;
	}

	return;
}