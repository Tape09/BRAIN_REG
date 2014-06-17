
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

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <string>


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


void copy_img(ImageType::Pointer & from_img, ImageType::Pointer & to_img);
int count_brain_pix(ImageType::Pointer & img);
int diff_img(ImageType::Pointer & from_img, ImageType::Pointer & to_img, ImageType::Pointer & out_img);
int getCommonPix(ImageType::Pointer & in_img, ImageTypeUC::Pointer & common_map);
void getInverseTfm(ImageType::Pointer & fixed_image, TransformTypeBSpline::Pointer & tfm, TransformTypeDis::Pointer & itfm);
void getLabelImage(ImageType::Pointer & image, Image<unsigned char, 3>::Pointer & out_image, int n_classes);
void getBrain(ImageType::Pointer & image, Image<unsigned char, 3>::Pointer & out_image, int n_bins = 100);


int main(int argc, char *argv[]) {

	itk::TimeProbe totalTimeClock;
	totalTimeClock.Start();

	if (argc < 4) {
		cout << "Please specify:" << endl;
		cout << " Properties File" << endl;
		cout << " Output Directory" << endl;
		cout << " Input Directory/Directories" << endl;
		return EXIT_FAILURE;
	} 

	unsigned int n_pcomponents;
	//try {
	//	n_pcomponents = stoi(argv[1]);
	//} catch (exception & e) {
	//	cerr << e.what() << endl;
	//	cerr << "N Principal Components invalid" << endl;
	//	return EXIT_FAILURE;
	//}
	
	// READ PROPERTIES // HAS TO SET VECTOR<STRING> OF PROP NAMES AND VECTOR<VECTOR<DOUBLE>> OF PROP VALUES
	typedef itk::CSVArray2DFileReader<double> ReaderTypeCSV;
	ReaderTypeCSV::Pointer csv_reader = ReaderTypeCSV::New();

	csv_reader->SetFileName(argv[1]);
	csv_reader->SetFieldDelimiterCharacter(',');
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

	//

	string output_dir_path(argv[2]);

	vector<string> input_dir_paths;
	for (int i = 3; i < argc; ++i) {
		input_dir_paths.push_back(argv[i]);
	}

	vector<ImageType::Pointer> images;
	vector<TransformTypeDis::Pointer> inv_tfms;
	

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

	for (int d = 0; d < input_dir_paths.size(); ++d) {
		cout << "Processing directory " << input_dir_paths[d] << "- " << d + 1 << "/" << input_dir_paths.size() << endl;
		

		// DIRECTORY STUFF
		string input_img_dir_path(input_dir_paths[d] + "/images");
		string input_tfm_dir_path(input_dir_paths[d] + "/transforms");

		try {
			input_img_dir->Load(input_img_dir_path.c_str());
			input_tfm_dir->Load(input_tfm_dir_path.c_str());
		} catch (ExceptionObject & err) {
			cerr << err << endl;
			return EXIT_FAILURE;
		}

		if (input_img_dir->GetNumberOfFiles() == 0) {
			cerr << "Input Directory Invalid" << endl;
			return EXIT_FAILURE;
		}

		if (input_tfm_dir->GetNumberOfFiles() == 0) {
			cerr << "Input Directory Invalid" << endl;
			return EXIT_FAILURE;
		}
		// 

		string img_path;
		string tfm_path;
		string img_name;
		string tfm_name;
		string output_img_path;
		string output_tfm_path;
		Directory::Pointer test_file = Directory::New();
		for (int i = 2; i < input_img_dir->GetNumberOfFiles(); ++i) {

			img_name = input_img_dir->GetFile(i);
			cout << "Processing Image " << img_name << "- " << i - 1 << "/" << input_img_dir->GetNumberOfFiles() - 2 << endl;

			img_path = input_img_dir_path + "/" + img_name;
			test_file->Load(img_path.c_str());
			

			if (test_file->GetNumberOfFiles() > 0) continue;

			int pos;
			pos = img_name.find_first_of(".");
			tfm_name = img_name.substr(0, pos) + ".tfm";
			tfm_path = input_tfm_dir_path + "/" + tfm_name;
			

			try {
				img_reader->SetFileName(img_path);
				img_reader->Update();
				duplicator->SetInputImage(img_reader->GetOutput());
				duplicator->Update();
				images.push_back(duplicator->GetOutput());

				tfm_reader->SetFileName(tfm_path);
				tfm_reader->Update();

				TransformTypeBSpline::Pointer tfm = static_cast<TransformTypeBSpline*>(tfm_reader->GetTransformList()->back().GetPointer());
				TransformTypeDis::Pointer itfm = TransformTypeDis::New();
				
				getInverseTfm(images.back(), tfm, itfm);

				inv_tfms.push_back(itfm);
			} catch (ExceptionObject & err) {
				cerr << err << endl;
				return EXIT_FAILURE;
			}			
		}
	}

	if (images.size() < 2) {
		cerr << "NOT ENOUGH IMAGES" << endl;
		return EXIT_FAILURE;
	}

	//duplicator->SetInputImage(images[0]);
	//duplicator->Update();

	ImageTypeUC::Pointer common_map = ImageTypeUC::New();

	//getLabelImage(images[0],common_map,k);
	getBrain(images[0], common_map);

	//ImageType::Pointer common_map = ImageType::New();
	//common_map = duplicator->GetOutput();	
	
	cout << "Calculating common pixels..." << endl;

	int n_common;
	for (int i = 1; i < images.size(); ++i) {
		n_common = getCommonPix(images[i], common_map);
		cout << "n common pixels: " << n_common << endl;
	}

	WriterTypeUC::Pointer mask_writer = WriterTypeUC::New();
	string mask_fn = output_dir_path + "/" + "common_mask.nii";
	mask_writer->SetFileName(mask_fn);
	mask_writer->SetInput(common_map);
	mask_writer->Update();


	vector<ImageTypePCA::Pointer> images_PCA(images.size());
	ImageTypePCA::RegionType region;
	itk::Size<2> sz;
	sz[0] = 3;
	sz[1] = n_common;
	region.SetSize(sz);

	for (int i = 0; i < images.size(); ++i) {
		images_PCA[i] = ImageTypePCA::New();
		images_PCA[i]->SetRegions(region);
		images_PCA[i]->Allocate();

		itk::ImageRegionIteratorWithIndex<ImageType> img_itr(images[i], images[i]->GetBufferedRegion());
		itk::ImageRegionIteratorWithIndex<ImageTypeUC> common_itr(common_map, common_map->GetBufferedRegion());
		itk::ImageRegionIteratorWithIndex<ImageTypePCA> pcaimg_itr(images_PCA[i], images_PCA[i]->GetBufferedRegion());

		double common_val;
		ImageType::IndexType img_idx;
		Point<double, 3> img_point;
		while (!img_itr.IsAtEnd()) {
			common_val = common_itr.Get();			
			if (common_val > 0) {
				img_idx = img_itr.GetIndex();
				images[i]->TransformIndexToPhysicalPoint(img_idx, img_point);
				img_point = inv_tfms[i]->TransformPoint(img_point);
				pcaimg_itr.Set(img_point[0]);
				++pcaimg_itr;
				pcaimg_itr.Set(img_point[1]);
				++pcaimg_itr;
				pcaimg_itr.Set(img_point[2]);
				++pcaimg_itr;
			}
			++img_itr;
			++common_itr;
		}
	}

	n_pcomponents = uint(images_PCA.size());

	typedef itk::ImagePCAShapeModelEstimator<ImageTypePCA, ImageTypePCA >  EstimatorType;
	EstimatorType::Pointer estimator = EstimatorType::New();
	estimator->SetNumberOfTrainingImages(images_PCA.size());
	estimator->SetNumberOfPrincipalComponentsRequired(n_pcomponents);

	for (int i = 0; i < images_PCA.size(); ++i) {
		estimator->SetInput(i, images_PCA[i]);
	}

	itk::TimeProbe clock;
	clock.Start();
	cout << "Preforming PCA analysis with " << n_pcomponents << " components..." << endl;
	estimator->Update();
	clock.Stop();
	cout << "Done!" << endl;
	cout << "Time taken: " << clock.GetTotal() << "s" << endl;

	
	string fn;
	fn = output_dir_path + "/mean.mhd";
	WriterTypePCA::Pointer pca_writer = WriterTypePCA::New();
	pca_writer->SetFileName(fn);
	pca_writer->SetInput(estimator->GetOutput(0));

	try {
		pca_writer->Update();
	} catch (ExceptionObject & err) {
		cerr << err << endl;
		return EXIT_FAILURE;
	}
	
	for (int i = 1; i < n_pcomponents+1; ++i) {
		fn = output_dir_path + "/PC_" + to_string(i) + ".mhd";
		pca_writer->SetFileName(fn);
		pca_writer->SetInput(estimator->GetOutput(i));
		try {
			pca_writer->Update();
		} catch (ExceptionObject & err) {
			cerr << err << endl;
			return EXIT_FAILURE;
		}
	}

	// PROPERTIES

	// set up projector
	ProjectorType::Pointer projector = ProjectorType::New();
	projector->SetBasisFromModel(estimator);

	// set up matrix
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A;
	A.resize(n_pcomponents, n_pcomponents);

	Eigen::Matrix<double, Eigen::Dynamic, 1> b;
	b.resize(n_pcomponents, 1);

	Eigen::Matrix<double, Eigen::Dynamic, 1> x;
	x.resize(n_pcomponents, 1);

	ProjectorType::BasisVectorType projection;

	for (int img_idx = 0; img_idx < images_PCA.size(); ++img_idx) {

		projector->SetImage(images_PCA[img_idx]);
		projector->Compute();
		projection = projector->GetProjection();
		//cout << projection.size() << endl;
		for (int i = 0; i < projection.size(); ++i) {
			A(img_idx, i) = projection[i];
		}

	}

	ofstream outfile;
	outfile.open(output_dir_path + "/properties.txt");
	for (int n = 0; n < prop_names.size(); ++n) {
		outfile << prop_names[n] << " ";
		
		for (int j = 0; j < prop_vals[n].size(); ++j) {
			b(j, 1) = prop_vals[n][j];
		}		

		x = A.colPivHouseholderQr().solve(b);

		for (int i = 0; i < n_pcomponents; ++i) {
			outfile << x(i, 1) << " ";
		}
	}
	outfile.close();
	
	totalTimeClock.Stop();
	cout << "ALL DONE!" << endl;
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