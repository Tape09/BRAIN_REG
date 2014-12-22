
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
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkMaskImageFilter.h"
#include "vnl/vnl_matrix.h"


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


//typedef itk::ImageFileWriter< ImageTypePCAtest > WriterTypePCAtest;

typedef ImageType::IndexType ImageIndexType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileReader<ImageTypeUC> ReaderTypeUC;
typedef itk::ImageFileWriter< ImageType  > WriterType;
typedef itk::ImageFileWriter< ImageTypePCA > WriterTypePCA;
typedef itk::ImageFileWriter< ImageTypeUC > WriterTypeUC;
typedef itk::ImageDuplicator< ImageType > DuplicatorType;
typedef itk::ImageDuplicator< ImageTypeUC > DuplicatorTypeUC;

using namespace std;
using namespace itk;


void copy_img(ImageType::Pointer & from_img, ImageType::Pointer & to_img);
int count_brain_pix(ImageType::Pointer & img);
int diff_img(ImageType::Pointer & from_img, ImageType::Pointer & to_img, ImageType::Pointer & out_img);
int getCommonPix(ImageType::Pointer & in_img, ImageTypeUC::Pointer & common_map);
void getInverseTfm(ImageTypeUC::Pointer & fixed_image, TransformTypeBSpline::Pointer & tfm, TransformTypeDis::Pointer & itfm);
void getLabelImage(ImageType::Pointer & image, Image<unsigned char, 3>::Pointer & out_image, int n_classes);
void getBrain(ImageType::Pointer & image, Image<unsigned char, 3>::Pointer & out_image, int n_bins = 100);
void txt_2_vnl_mat(string fn, vnl_matrix<double> & mat);
void vnl_mat_2_txt(string fn, vnl_matrix<double> & mat);

int main(int argc, char *argv[]) {

	//cout.precision(15);


	itk::TimeProbe totalTimeClock;
	totalTimeClock.Start();

	if (argc < 5) {
		cout << "Please specify:" << endl;
		cout << " Properties File" << endl;
		cout << " Input Directory/Directories" << endl;
		//cout << " Significant Pixels directory" << endl;
		cout << " Output Directory" << endl;
		cout << " Projections file" << endl;
		cout << " [n_pcomponents]" << endl;
		//cout << " [randomize seed]" << endl;
		return EXIT_FAILURE;
	}

	unsigned int n_pcomponents;
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
	string fn_proj_file(argv[4]);
	if (argc >= 6) {
		n_pcomponents = stoi(argv[5]);		
	} else {
		n_pcomponents = n_images;
	}

	cout << "n_pcomponents: " << n_pcomponents << endl;
	cout << "n_input_images: " << n_images << endl;
	if (n_pcomponents > n_images || n_pcomponents < 2) {
		cerr << "Error: n_pcomponents > n_input_images; or n_pcomponents < 2" << endl;
		exit(EXIT_FAILURE);
	}

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

	ReaderTypeUC::Pointer seg_reader = ReaderTypeUC::New();
	string seg_fn = input_dir_path + "/seg3.nii.gz";
	//cout << seg_fn << endl;
	seg_reader->SetFileName(seg_fn);
	seg_reader->Update();
	ImageTypeUC::Pointer seg_img = seg_reader->GetOutput();
	itk::ImageRegionIteratorWithIndex<ImageTypeUC> seg_itr = itk::ImageRegionIteratorWithIndex<ImageTypeUC>(seg_img, seg_img->GetBufferedRegion());
	while (!seg_itr.IsAtEnd()) {
		int val = seg_itr.Get();
		if (val <= 1)
			seg_itr.Set(0);
		else
			seg_itr.Set(1);

		++seg_itr;
	}

	typedef itk::MaskImageFilter< ImageTypeUC, ImageTypeUC > MaskFilterType;
	MaskFilterType::Pointer maskFilter = MaskFilterType::New();
	maskFilter->SetInput(common_map);
	maskFilter->SetMaskImage(seg_img);
	maskFilter->Update();
	common_map = maskFilter->GetOutput();

	// ______________________________________________________________________________________________________________________________________
	typedef itk::BinaryBallStructuringElement<ImageTypeUC::PixelType, 3> StructuringElementType;
	StructuringElementType structuringElement;
	structuringElement.SetRadius(1.0);
	structuringElement.CreateStructuringElement();

	typedef itk::BinaryErodeImageFilter <ImageTypeUC, ImageTypeUC, StructuringElementType> BinaryErodeImageFilterType;
	BinaryErodeImageFilterType::Pointer erodeFilter = BinaryErodeImageFilterType::New();
	erodeFilter->SetInput(common_map);
	erodeFilter->SetKernel(structuringElement);
	erodeFilter->SetErodeValue(1);
	erodeFilter->Update();

	common_map = erodeFilter->GetOutput();
	// _______________________________________________________________________________________________________________________________________

	WriterTypeUC::Pointer writer_uc = WriterTypeUC::New();
	string common_fn = output_dir_path + "/common_mask.nii";
	writer_uc->SetFileName(common_fn);
	writer_uc->SetInput(common_map);
	writer_uc->Update();

	//return 0;


	//cout << "asfd" << endl;
	vector<ImageTypePCA::Pointer> images_PCA(n_images);
	ImageTypePCA::RegionType region;
	itk::Size<2> sz;
	sz[0] = 3;
	sz[1] = n_common;
	region.SetSize(sz);

	cout << endl;
	for (int i = 0; i < n_images; ++i) {
		string img_name = fn_prefix + img_names[i] + fn_img_suffix;
		string img_path = input_img_dir_path + "/" + img_name;
		cout << "Processing Image " << i + 1 << "/" << tfms.size() << endl;
		img_reader->SetFileName(img_path);
		img_reader->Update();

		ImageType::Pointer temp_img = img_reader->GetOutput();

		images_PCA[i] = ImageTypePCA::New();
		images_PCA[i]->SetRegions(region);
		images_PCA[i]->Allocate();

		itk::ImageRegionIteratorWithIndex<ImageTypeUC> common_itr(common_map, common_map->GetBufferedRegion());
		itk::ImageRegionIteratorWithIndex<ImageType> img_itr(temp_img, temp_img->GetBufferedRegion());
		itk::ImageRegionIteratorWithIndex<ImageTypePCA> pcaimg_itr(images_PCA[i], images_PCA[i]->GetBufferedRegion());

		unsigned int common_val;
		Point<double, 3> com_point;
		Point<double, 3> aff_point;
		ImageType::IndexType img_idx;
		Point<double, 3> img_point;
		while (!common_itr.IsAtEnd()) {
			common_val = common_itr.Get();			
			if (common_val > 0.5) {
				img_idx = common_itr.GetIndex();
				
				temp_img->TransformIndexToPhysicalPoint(img_idx, com_point);
				aff_point = tfms[i]->TransformPoint(com_point);
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
	}

	
	for (int q = 0; q < tfms.size(); ++q) {
		//tfms[q]->Delete();
		tfms[q] = NULL;
	}
	tfms.clear();

	typedef itk::ImagePCAShapeModelEstimator<ImageTypePCA, ImageTypePCA >  EstimatorType;
	typedef itk::ImagePCADecompositionCalculator<ImageTypePCA, ImageTypePCA> ProjectorType;
	ProjectorType::BasisImagePointerVector basis_image_vector;
	ProjectorType::BasisImagePointer mean_image;

	{
		EstimatorType::Pointer estimator = EstimatorType::New();
		estimator->SetNumberOfTrainingImages(n_images);
		estimator->SetNumberOfPrincipalComponentsRequired(n_pcomponents);

		for (int i = 0; i < images_PCA.size(); ++i) {
			estimator->SetInput(i, images_PCA[i]);
		}

		itk::TimeProbe clock;
		clock.Start();
		cout << "Performing PCA analysis with " << n_pcomponents << " components..." << endl;
		estimator->Update();
		clock.Stop();
		cout << "Time taken: " << clock.GetTotal() << "s" << endl;

		cout << "Printing..." << endl;
		string fn;
		fn = output_dir_path + "/mean.mhd";
		WriterTypePCA::Pointer pca_writer = WriterTypePCA::New();
		pca_writer->SetFileName(fn);
		pca_writer->SetInput(estimator->GetOutput(0));
		mean_image = estimator->GetOutput(0);

		try {
			pca_writer->Update();
		} catch (ExceptionObject & err) {
			cerr << err << endl;
			return EXIT_FAILURE;
		}

		for (int i = 1; i < n_pcomponents + 1; ++i) {
			fn = output_dir_path + "/PC_" + to_string(i) + ".mhd";
			pca_writer->SetFileName(fn);
			pca_writer->SetInput(estimator->GetOutput(i));
			basis_image_vector.push_back(estimator->GetOutput(i));
			try {
				pca_writer->Update();
			} catch (ExceptionObject & err) {
				cerr << err << endl;
				return EXIT_FAILURE;
			}
		}
	}

	

	

	

	//estimator->Delete();
	//estimator = NULL;
	// PROPERTIES
	cout << "Calculating projections..." << endl;

	// set up projector
	ProjectorType::Pointer projector = ProjectorType::New();

	projector->SetMeanImage(mean_image);
	projector->SetBasisImages(basis_image_vector);
	
	//for (int q = 0; q < images_PCA.size(); ++q) {
	//	images_PCA[q] = NULL;
	//}

	


	ProjectorType::BasisVectorType projection;

	vnl_matrix<double> projs_mat;
	projs_mat.set_size(n_images, n_pcomponents+1);
	for (int j = 0; j < n_images; ++j) {
		projector->SetImage(images_PCA[j]);
		projector->Compute();
		projection = projector->GetProjection();
		for (int i = 0; i < projection.size(); ++i) {
			projs_mat(j, i + 1) = projection[i];
		}
		cout << "Progress: " << j * 100 / n_images << "%\r";

	}

	for (int j = 0; j < img_names.size(); ++j) {
		projs_mat(j, 0) = data->GetData(img_names[j], prop_names[0]);
	}

	vnl_mat_2_txt(fn_proj_file, projs_mat);

	totalTimeClock.Stop();
	cout << "ALL DONE!!!!!" << endl;
	cout << "Total time taken: " << totalTimeClock.GetTotal() << "s" << endl;

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


void getInverseTfm(ImageTypeUC::Pointer & fixed_image, TransformTypeBSpline::Pointer & tfm, TransformTypeDis::Pointer & itfm) {
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


	//cout << "bigbin: " << bigbin << endl;
	//cout << "lower: " << lower_bound << endl;
	//cout << "upper: " << upper_bound << endl;
	//cout << "ratio: " << back_ratio << endl;
	//cout << "change: 3" << endl;

	//DuplicatorType::Pointer duplicator = DuplicatorType::New();
	//duplicator->SetInputImage(image);
	//duplicator->Update();

	out_image->SetRegions(image->GetBufferedRegion());
	out_image->SetDirection(image->GetDirection());
	out_image->SetOrigin(image->GetOrigin());
	out_image->SetSpacing(image->GetSpacing());
	out_image->Allocate();

	//out_image = duplicator->GetOutput();


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

void txt_2_vnl_mat(string fn, vnl_matrix<double> & mat) {
	ifstream txtfile(fn);
	txtfile.precision(25);
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
				mat(r, c) = atof(val.c_str());
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
			//cout << mat(r, c) << " ";
		}
		outfile << endl;
		//cout << endl;
	}

	outfile.close();
}