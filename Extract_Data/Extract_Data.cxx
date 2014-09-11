
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
#include "itkImageToHistogramFilter.h"
#include "itkMedianImageFilter.h"
#include "itkCSVArray2DFileReader.h"
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_matlab_filewrite.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <string>
#include <cstdlib>
#include <algorithm>

typedef itk::Image< double, 3 > ImageType;
typedef itk::Image< unsigned char, 3 > ImageTypeUC;
typedef itk::Image< double, 2 > ImageTypePCA;

typedef unsigned uint;

const int k = 20;

const unsigned int spline_order = 3;
typedef itk::BSplineTransform<double, 3, spline_order> TransformTypeBSpline;

typedef itk::Vector<double, 3> VectorType;
typedef itk::Point<double, 3> PointType;
//typedef itk::Image< PointType, 2 > ImageTypePCA;
typedef itk::Image< VectorType, 3 > DisplacementFieldType;
typedef itk::DisplacementFieldTransform<double, 3> TransformTypeDis;
typedef itk::ImagePCADecompositionCalculator<ImageTypePCA, ImageTypePCA> ProjectorType;

//typedef itk::ImageFileWriter< ImageTypePCAtest > WriterTypePCAtest;

typedef ImageType::IndexType ImageIndexType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter< ImageType  > WriterType;
typedef itk::ImageFileWriter< ImageTypePCA > WriterTypePCA;
typedef itk::ImageFileWriter< ImageTypeUC > WriterTypeUC;
typedef itk::ImageDuplicator< ImageType > DuplicatorType;

using namespace std;
using namespace itk;



int getCommonPix(ImageType::Pointer & in_img, ImageTypeUC::Pointer & common_map);
void getBrain(ImageType::Pointer & image, Image<unsigned char, 3>::Pointer & out_image, int n_bins = 100);


int main(int argc, char *argv[]) {

	//cout.precision(15);


	itk::TimeProbe totalTimeClock;
	totalTimeClock.Start();

	if (argc < 4) {
		cout << "Please specify:" << endl;
		cout << " Properties File" << endl;
		cout << " Input Directory/Directories" << endl;
		cout << " Output Directory" << endl;		
		//cout << " [n_pcomponents]" << endl;
		//cout << " [randomize seed]" << endl;
		return EXIT_FAILURE;
	} 

	//unsigned int n_pcomponents;
	unsigned int n_images;


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

	n_images = img_names.size();

	string output_dir_path(argv[3]);

	string input_dir_path(argv[2]);
	string input_img_dir_path(input_dir_path + "/images");
	string input_tfm_dir_path(input_dir_path + "/transforms");

	cout << "n_input_images: " << n_images << endl;

	vector<TransformTypeDis::Pointer> inv_tfms;
	vector<TransformTypeBSpline::Pointer> tfms;
	

	Directory::Pointer input_img_dir = Directory::New();
	Directory::Pointer input_tfm_dir = Directory::New();
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

	itk::TransformFileReader::Pointer tfm_reader = itk::TransformFileReader::New();
	ReaderType::Pointer img_reader = ReaderType::New();
	WriterType::Pointer img_writer = WriterType::New();
	DuplicatorType::Pointer duplicator = DuplicatorType::New();

	int n_common;
	bool gotBrain = false;
	ImageTypeUC::Pointer common_map = ImageTypeUC::New();
	for (int i = 0; i < n_images; ++i) {
		string img_path;
		string tfm_path;
		string img_name;
		string tfm_name;
		string output_img_path;
		string output_tfm_path;

		img_name = fn_prefix + img_names[i] + fn_img_suffix;
		img_path = input_img_dir_path + "/" + img_name;
		cout << "Reading Image " << img_name << " - " << i + 1 << "/" << img_names.size() << endl;
		

		tfm_name = fn_prefix + img_names[i] + fn_tfm_suffix;
		tfm_path = input_tfm_dir_path + "/" + tfm_name;

		try {
			img_reader->SetFileName(img_path);
			img_reader->Update();

			ImageType::Pointer temp_img = img_reader->GetOutput();

			if (!gotBrain) {
				getBrain(temp_img, common_map);
				gotBrain = true;
			} else {
				n_common = getCommonPix(temp_img, common_map);
			}

			tfm_reader->SetFileName(tfm_path);
			tfm_reader->Update();

			TransformTypeBSpline::Pointer tfm = static_cast<TransformTypeBSpline*>(tfm_reader->GetTransformList()->back().GetPointer());

			tfms.push_back(tfm);

		} catch (ExceptionObject & err) {
			cerr << err << endl;
			return EXIT_FAILURE;
		}
	}


	if (tfms.size() < 2) {
		cerr << "NOT ENOUGH IMAGES" << endl;
		return EXIT_FAILURE;
	}

	WriterTypeUC::Pointer mask_writer = WriterTypeUC::New();
	string mask_fn = output_dir_path + "/" + "common_mask.nii";
	mask_writer->SetFileName(mask_fn);
	mask_writer->SetInput(common_map);
	mask_writer->Update();

	//cout << "asfd" << endl;
	vector<ImageTypePCA::Pointer> images_PCA(tfms.size());
	ImageTypePCA::RegionType region;
	itk::Size<2> sz;
	sz[0] = 3;
	sz[1] = n_common;
	region.SetSize(sz);

	cout << endl;

	

	for (int i = 0; i < tfms.size(); ++i) {
		vnl_matrix<double> mat(n_common,4);

		string img_path;
		string img_name;
		string output_img_path;		

		img_name = fn_prefix + img_names[i] + fn_img_suffix;
		img_path = input_img_dir_path + "/" + img_name;

		img_reader->SetFileName(img_path);
		img_reader->Update();

		ImageType::Pointer temp_img = img_reader->GetOutput();

		cout << "Processing Image " << i + 1 << "/" << tfms.size() << endl;

		images_PCA[i] = ImageTypePCA::New();
		images_PCA[i]->SetRegions(region);
		images_PCA[i]->Allocate();

		itk::ImageRegionIteratorWithIndex<ImageTypeUC> common_itr(common_map, common_map->GetBufferedRegion());
		itk::ImageRegionIteratorWithIndex<ImageType> img_itr(temp_img, temp_img->GetBufferedRegion());

		unsigned int common_val;
		ImageType::IndexType img_idx;
		Point<double, 3> img_point;
		int p = 0;
		while (!common_itr.IsAtEnd()) {
			common_val = common_itr.Get();			
			if (common_val > 0.5) {
				img_idx = img_itr.GetIndex();
				temp_img->TransformIndexToPhysicalPoint(img_idx, img_point);
				double img_val = img_itr.Get();
				img_point = tfms[i]->TransformPoint(img_point);

				mat(p, 0) = img_point[0];
				mat(p, 1) = img_point[1];
				mat(p, 2) = img_point[2];
				mat(p, 3) = img_val;
				++p;					
			}
			++common_itr;
			++img_itr;
		}

		string fn = output_dir_path + "/" + fn_prefix + img_names[i] + ".mat";
		vnl_matlab_filewrite mat_writer(fn.c_str());
		mat_writer.write(mat);
	}


		
	totalTimeClock.Stop();
	cout << "ALL DONE!!!!!" << endl;
	cout << "Total time taken: " << totalTimeClock.GetTotal() << "s" << endl;

	return EXIT_SUCCESS;
}



int getCommonPix(ImageType::Pointer & in_img, ImageTypeUC::Pointer & common_map) {

	ImageTypeUC::Pointer label_image = ImageTypeUC::New();
	//getLabelImage(in_img, label_image, k);
	getBrain(in_img, label_image);

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

void getBrain(ImageType::Pointer & image, Image<unsigned char, 3>::Pointer & out_image, int n_bins) {
	typedef itk::Statistics::ImageToHistogramFilter<ImageType > HistogramFilterType;
	HistogramFilterType::Pointer histogramFilter = HistogramFilterType::New();

	typedef HistogramFilterType::HistogramSizeType SizeType;
	SizeType size(1);
	size[0] = n_bins;

	histogramFilter->SetHistogramSize(size);


	HistogramFilterType::HistogramMeasurementVectorType lowerBound(1);
	HistogramFilterType::HistogramMeasurementVectorType upperBound(1);

	typedef MinimumMaximumImageCalculator<ImageType> imageCalculatorType;
	imageCalculatorType::Pointer calculator = imageCalculatorType::New();
	calculator->SetImage(image);
	calculator->Compute();
	const double minIntensity = calculator->GetMinimum();
	const double maxIntensity = calculator->GetMaximum();

	lowerBound[0] = calculator->GetMinimum();
	upperBound[0] = calculator->GetMaximum();

	histogramFilter->SetHistogramBinMinimum(lowerBound);
	histogramFilter->SetHistogramBinMaximum(upperBound);

	histogramFilter->SetInput(image);
	try {
		histogramFilter->Update();
	} catch (itk::ExceptionObject e) {
		cerr << e << endl;
		exit(EXIT_FAILURE);
	}


	typedef HistogramFilterType::HistogramType HistogramType;
	HistogramType * histogram = histogramFilter->GetOutput();

	double total = histogram->GetTotalFrequency();

	int max_val = 0;
	int max_idx = 0;

	for (unsigned int i = 0; i < histogram->GetSize()[0]; ++i) {
		int val = histogram->GetFrequency(i);

		if (val > max_val) {
			max_val = val;
			max_idx = i;
		}
	}

	double gap = (maxIntensity - minIntensity) / (n_bins);
	double bigbin = minIntensity - gap / 2 + gap*(max_idx + 1);

	double lower_bound = bigbin - gap / 2;
	double upper_bound = bigbin + gap / 2;

	double back_ratio = max_val / total;

	int l_idx = max_idx;
	int u_idx = max_idx;
	int back_freq = max_val;
	while (back_ratio < 0.5) {
		if (l_idx == 0) {
			++u_idx;
			upper_bound += gap;
			back_freq += histogram->GetFrequency(u_idx);
		} else if (u_idx == histogram->GetSize()[0] - 1) {
			--l_idx;
			lower_bound -= gap;
			back_freq += histogram->GetFrequency(l_idx);
		} else {
			if (histogram->GetFrequency(l_idx - 1) > histogram->GetFrequency(u_idx + 1)) {
				--l_idx;
				lower_bound -= gap;
				back_freq += histogram->GetFrequency(l_idx);
			} else {
				++u_idx;
				upper_bound += gap;
				back_freq += histogram->GetFrequency(u_idx);
			}
		}

		back_ratio = back_freq / total;
	}

	out_image->SetRegions(image->GetBufferedRegion());
	out_image->SetDirection(image->GetDirection());
	out_image->SetOrigin(image->GetOrigin());
	out_image->SetSpacing(image->GetSpacing());
	out_image->Allocate();

	itk::ImageRegionIteratorWithIndex<ImageType> imgIterator(image, image->GetBufferedRegion());
	itk::ImageRegionIteratorWithIndex<ImageTypeUC> outimgIterator(out_image, out_image->GetBufferedRegion());
	while (!imgIterator.IsAtEnd()) {

		double val = imgIterator.Get();

		if (lower_bound <= val && val <= upper_bound) {
			outimgIterator.Set(0);
		} else {
			outimgIterator.Set(1);
		}

		++imgIterator;
		++outimgIterator;
	}


	typedef itk::MedianImageFilter<ImageTypeUC, ImageTypeUC > FilterType;
	FilterType::Pointer medianFilter = FilterType::New();
	FilterType::InputSizeType radius;
	radius.Fill(1);

	medianFilter->SetRadius(radius);
	medianFilter->SetInput(out_image);
	medianFilter->Update();

	out_image = medianFilter->GetOutput();

}