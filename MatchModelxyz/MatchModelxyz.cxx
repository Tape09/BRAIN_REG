
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
void txt_2_vnl_mat(string fn, vnl_matrix<double> & mat);
void vnl_mat_2_txt(string fn, vnl_matrix<double> & mat);

int main(int argc, char *argv[]) {

	itk::TimeProbe totalTimeClock;
	totalTimeClock.Start();

	if (argc < 6) {
		cout << "Please specify:" << endl;
		cout << " CSV File" << endl;
		cout << " Input folder" << endl;
		cout << " Input PCA folder" << endl;
		cout << " Output folder" << endl;
		cout << " Output proj file" << endl;
		//cout << " [Reconstruct]" << endl;
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
	string output_fname(argv[5]);
	bool reconstruct_images = false;
	//if (argc == 7) reconstruct_images = true;

	//vnl_matrix<double> norm_factors_mat;
	//txt_2_vnl_mat(input_pca_path + "/norm_factors.txt", norm_factors_mat);

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

	int n_pcomponents = basis_image_vector.size();
	cout << "n_pcomponents: " << n_pcomponents << endl;



	ofstream outfile;
	string out_fn = output_fname;
	outfile.precision(25);
	outfile.open(out_fn, ios::trunc);

	for (int i = 0; i < img_names.size(); ++i) {
		// READ IMAGE
		string img_name = fn_prefix + img_names[i] + fn_img_suffix;
		string img_path = input_img_dir + "/images/" + img_name;

		cout << "Processing Image " << img_name << "- " << i + 1 << "/" << img_names.size() << endl;

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
		//duplicator->SetInputImage(input_image);
		//duplicator->Update();
		//ImageType::Pointer input_image_copy = ImageType::New();
		//input_image_copy = duplicator->GetOutput();

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
		itk::ImageRegionIteratorWithIndex<ImageType> img_itr(input_image, input_image->GetBufferedRegion());
		itk::ImageRegionIteratorWithIndex<ImageTypePCA> pcaimg_itr(input_PCA_image, input_PCA_image->GetBufferedRegion());

		unsigned int common_val;
		Point<double, 3> com_point;
		Point<double, 3> aff_point;
		ImageType::IndexType img_idx;
		Point<double, 3> img_point;
		while (!common_itr.IsAtEnd()) {
			common_val = common_itr.Get();			
			if (common_val > 0.5) {
				img_idx = common_itr.GetIndex();
				
				input_image->TransformIndexToPhysicalPoint(img_idx, com_point);
				aff_point = tfm->TransformPoint(com_point);
				double img_val = img_itr.Get();
				

				pcaimg_itr.Set(com_point[0] - aff_point[0]);
				++pcaimg_itr;
				pcaimg_itr.Set(com_point[1] - aff_point[1]);
				++pcaimg_itr;
				pcaimg_itr.Set(com_point[2] - aff_point[2]);
				++pcaimg_itr;						
			}
			++common_itr;
			++img_itr;
		}	


		// PLUG IT ALL IN
		projector->SetImage(input_PCA_image);
		projector->SetMeanImage(mean_image);
		projector->SetBasisImages(basis_image_vector);

		projector->Compute();

		ProjectorType::BasisVectorType projection = projector->GetProjection();
		outfile << data->GetData(img_names[i], prop_names[0]) << " ";
		
		for (int i = 0; i < projection.size(); ++i) {
			outfile << projection[i] << " ";
		}
		outfile << endl;

	}
	outfile.close();

	cout << "ALL DONE!" << endl;
	return EXIT_SUCCESS;
	
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

void txt_2_vnl_mat(string fn, vnl_matrix<double> & mat) {
	ifstream txtfile(fn);
	if (txtfile.is_open()) {
		int p_idx = 0;
		string line;
		string val;
		{
			getline(txtfile, line);
			istringstream sss(line);
			getline(sss, val, ' ');
			int nr = atoi(val.c_str());
			//cout << "nr=" << nr << endl;
			getline(sss, val, ' ');
			int nc = atoi(val.c_str());
			//cout << "nc=" << nc << endl;
			mat.set_size(nr, nc);
		}

		int r = 0;
		while (getline(txtfile, line)) {
			istringstream ss(line);
			int c = 0;
			while (getline(ss, val, ' ')) {
				mat(r, c) = atoi(val.c_str());
				//cout << "asdf:" << val << endl;
				++c;
			}
			++r;
		}
		txtfile.close();
	}
}

void vnl_mat_2_txt(string fn, vnl_matrix<double> & mat) {
	ofstream outfile;
	outfile.precision(25);
	outfile.open(fn);

	outfile << mat.rows() << " " << mat.cols() << endl;

	for (int r = 0; r < mat.rows(); ++r) {
		for (int c = 0; c < mat.cols(); ++c) {
			outfile << mat(r, c) << " ";
		}
		outfile << endl;
	}

	outfile.close();
}