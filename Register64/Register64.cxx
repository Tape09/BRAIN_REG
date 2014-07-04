#include "itkResampleImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageDuplicator.h"
#include "itkDirectory.h"
#include "itkFileTools.h"
#include "itkExceptionObject.h"
#include "itkSimilarity3DTransform.h"
#include "itkImageRegistrationMethod.h"
#include "itkMutualInformationImageToImageMetric.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNormalizeImageFilter.h"
#include "itkAffineTransform.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkCenteredTransformInitializer.h"
#include "itkMultiResolutionImageRegistrationMethod.h"
#include "itkTimeProbe.h"
#include "itkCommand.h"
#include "itkBSplineTransform.h"
#include "itkLBFGSBOptimizer.h"
#include "itkTransformFileWriter.h"
#include "itkLBFGSOptimizer.h"
#include "itkScalarImageKmeansImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkCastImageFilter.h"
#include "itkImageToHistogramFilter.h"
#include "itkMedianImageFilter.h"
#include "itkDisplacementFieldTransform.h"
#include "itkIterativeInverseDisplacementFieldImageFilter.h"

#include <deque>
#include <math.h>
#include <iostream>
#include <string>


typedef itk::Image< double, 3 > ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter< ImageType  > WriterType;
typedef itk::ImageDuplicator< ImageType > DuplicatorType;

typedef itk::Vector<double, 3> VectorType;
typedef itk::Point<double, 3> PointType;
typedef itk::Image< VectorType, 3 > DisplacementFieldType;

const unsigned int spline_order = 3;
typedef itk::BSplineTransform<double, 3, spline_order> TransformTypeBSpline;
typedef itk::AffineTransform<double, 3> TransformTypeAffine;
typedef itk::DisplacementFieldTransform<double, 3> TransformTypeDis;

const int k = 20;


using namespace itk;
using namespace std;

template <typename TRegistration>
class RegistrationInterfaceCommand : public itk::Command {
public:
	typedef  RegistrationInterfaceCommand   Self;
	typedef  itk::Command                   Superclass;
	typedef  itk::SmartPointer<Self>        Pointer;
	itkNewMacro(Self);

	double min_step;

protected:
	RegistrationInterfaceCommand() {};

public:
	typedef   TRegistration                              RegType;
	typedef   RegType *									 RegistrationPointer;
	typedef   itk::RegularStepGradientDescentOptimizer   OptimizerType;
	typedef   OptimizerType *                            OptimizerPointer;

	void Execute(itk::Object * object, const itk::EventObject & event) {
		if (!(itk::IterationEvent().CheckEvent(&event))) {
			return;
		}

		RegistrationPointer registration = dynamic_cast<RegistrationPointer>(object);

		OptimizerPointer optimizer = dynamic_cast< OptimizerPointer >(registration->GetModifiableOptimizer());

		std::cout << std::endl;
		std::cout << "-= MultiResolution Level " << registration->GetCurrentLevel() << " =-" << std::endl;		
		if (registration->GetCurrentLevel() == 0) {
			min_step = optimizer->GetMinimumStepLength();
			optimizer->SetMinimumStepLength(0.1);
		} else {
			optimizer->SetMaximumStepLength(optimizer->GetMinimumStepLength() * 5.0);			
			optimizer->SetMinimumStepLength(min_step);					
		}
		
		cout << "Max step length: " << optimizer->GetMaximumStepLength() << endl;
		cout << "Min step length: " << optimizer->GetMinimumStepLength() << endl;
	}

	void Execute(const itk::Object *, const itk::EventObject &) {
		return; 
	}
};

class CommandIterationUpdate : public itk::Command {
public:
	typedef  CommandIterationUpdate   Self;
	typedef  itk::Command             Superclass;
	typedef itk::SmartPointer<Self>   Pointer;
	itkNewMacro(Self);

	double last_metric_val;
	int counter;
	deque<double> deq;

protected:
	CommandIterationUpdate() { 
		last_metric_val = -99999999; 
		counter = 0; 
	};

public:
	typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
	typedef   const OptimizerType *                  OptimizerPointerC;
	typedef   OptimizerType *                  OptimizerPointer;

	void Execute(itk::Object *caller, const itk::EventObject & event) {
		Execute((const itk::Object *)caller, event);	

		OptimizerPointer optimizer = dynamic_cast< OptimizerPointer >(caller);
		if (!itk::IterationEvent().CheckEvent(&event)) {
			return;
		}
		//
		//double diff = optimizer->GetGradient().two_norm();
		//if (diff < 0.03) {
		//	++counter;
		//	if (counter > 10) {
		//		optimizer->StopOptimization();
		//	}
		//} else {
		//	counter = 0;
		//}
		
		double diff = last_metric_val - optimizer->GetValue();
		deq.push_front(diff);
		if (deq.size() > 10) deq.pop_back();

		double avg = 0;
		for (auto it = deq.begin(); it != deq.end(); ++it) {
			avg += *it;
		}
		avg /= deq.size();

		cout << "avg: " << avg << endl;
		last_metric_val = optimizer->GetValue();

		if (avg < 5e-5 && optimizer->GetCurrentIteration() > 25) {
			++counter;
			//cout << "count: " << counter << endl;
			if (counter > 10) {
				optimizer->StopOptimization();
			}
		} else {
			counter = 0;
		}

		//cout << "diff: " << diff << endl;

	}

	void Execute(const itk::Object * object, const itk::EventObject & event) {
		OptimizerPointerC optimizer = dynamic_cast< OptimizerPointerC >(object);
		if (!itk::IterationEvent().CheckEvent(&event)) {
			return;
		}
		std::cout << "Iteration: " << optimizer->GetCurrentIteration() << "\t";
		std::cout << "Metric Value: " << optimizer->GetValue() << "\t";
		std::cout << "Step Length: " << optimizer->GetCurrentStepLength() << "\t";
		cout << "Gradient: " << optimizer->GetGradient().one_norm() << endl;
	}
};

class CommandIterationUpdateBSpline : public itk::Command {
public:
	typedef  CommandIterationUpdateBSpline   Self;
	typedef  itk::Command             Superclass;
	typedef itk::SmartPointer<Self>   Pointer;
	itkNewMacro(Self);

	double last_metric_val;
	int counter;

protected:
	CommandIterationUpdateBSpline() {
		last_metric_val = -99999999;
		counter = 0;
	};

public:
	typedef itk::LBFGSBOptimizer OptimizerType;
	typedef   const OptimizerType *                  OptimizerPointerC;
	typedef   OptimizerType *                  OptimizerPointer;

	void Execute(itk::Object *caller, const itk::EventObject & event) {
		Execute((const itk::Object *)caller, event);

		OptimizerPointer optimizer = dynamic_cast< OptimizerPointer >(caller);
		if (!itk::IterationEvent().CheckEvent(&event)) {
			return;
		}

		double diff = abs(last_metric_val - optimizer->GetValue());
		//cout << "diff: " << diff << endl;
		last_metric_val = optimizer->GetValue();
		if (diff < 0.0003) {
			++counter;			
			//cout << "count: " << counter << endl;
			if (counter > 10) {
				optimizer->SetMaximumNumberOfIterations(0);
				optimizer->SetMaximumNumberOfEvaluations(0);
			}
		} else {
			counter = 0;
		}		
	}

	void Execute(const itk::Object * object, const itk::EventObject & event) {
		OptimizerPointerC optimizer =
			dynamic_cast< OptimizerPointerC >(object);
		if (!itk::IterationEvent().CheckEvent(&event)) {
			return;
		}
		std::cout << "Iteration: " << optimizer->GetCurrentIteration() << "\t";
		std::cout << "Metric Value: " << optimizer->GetValue() << endl;		
		
		//std::cout << optimizer->GetCurrentPosition() << std::endl;

	}
};

void SmoothAndNormalize(ImageType::Pointer & input_image, ImageType::Pointer & output_image, double variance = 2.0);
void RegisterImages_MIAffine(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image, TransformTypeAffine::Pointer & out_transform);
void RegisterImages_MIAffineMulti(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image, TransformTypeAffine::Pointer & out_transform);
void RegisterImages_MSAffine(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image, TransformTypeAffine::Pointer & out_transform);
void RegisterImages_MIBSpline(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image, TransformTypeBSpline::Pointer & out_transform);
void RegisterImages_MSBSpline(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image, TransformTypeBSpline::Pointer & out_transform);

void getVolumeAndCenter(ImageType::Pointer & image, double & volume, vector<double> & center);
void changeBasis(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image);
void initTrans(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image);
void getBrain(ImageType::Pointer & image, ImageType::Pointer & out_image, int n_bins = 100);
void getLabelImage(ImageType::Pointer & image, ImageType::Pointer & out_image, int n_classes);
void getInverseTfm(const ImageType::Pointer & fixed_image, const TransformTypeBSpline::Pointer & tfm, TransformTypeDis::Pointer & itfm);

int main(int argc, char *argv[])
{
	if (argc < 6) {
		cerr << "INPUT FAIL" << endl;
		cout << "Usage: " << endl;
		cout << " Fixed Image" << endl;
		cout << " Moving Image" << endl;
		cout << " Output Image Name" << endl;
		cout << " Output Transform Name" << endl;
		cout << " Transform Type" << endl;
		return EXIT_FAILURE;
	}


// READ FILES BEGIN
	ReaderType::Pointer fixed_reader = ReaderType::New();
	ReaderType::Pointer moving_reader = ReaderType::New();
	fixed_reader->SetFileName(argv[1]);
	moving_reader->SetFileName(argv[2]);
	fixed_reader->Update();
	moving_reader->Update();
	
	ImageType::Pointer fixed_image = fixed_reader->GetOutput();
	ImageType::Pointer moving_image = moving_reader->GetOutput();	
// READ FILES END

// CHANGE BASIS AND COPY
	changeBasis(fixed_image, moving_image);
	typedef itk::ImageDuplicator<ImageType> DuplicatorType;
	DuplicatorType::Pointer duplicator = DuplicatorType::New();
	duplicator->SetInputImage(moving_image);
	duplicator->Update();
	ImageType::Pointer original_moving_image = duplicator->GetOutput();
	duplicator->SetInputImage(fixed_image);
	duplicator->Update();
	ImageType::Pointer original_fixed_image = duplicator->GetOutput();


// CHANGE BASIS AND COPY END
	
	typedef itk::ResampleImageFilter<ImageType, ImageType > ResampleFilterType;
	ResampleFilterType::Pointer resample = ResampleFilterType::New();

	TransformTypeBSpline::Pointer finalTransformBSpline;
	TransformTypeAffine::Pointer finalTransformAffine;

	ImageType::Pointer fixed_image_mask;
	ImageType::Pointer moving_image_mask;

	itk::TimeProbe clock;
	clock.Start();
	//bool bspline = false;
	if (strcmp(argv[5],"bspline") == 0) {	
		SmoothAndNormalize(fixed_image, fixed_image);
		SmoothAndNormalize(moving_image, moving_image);
		RegisterImages_MIBSpline(fixed_image, moving_image, finalTransformBSpline);
		resample->SetTransform(finalTransformBSpline);
		//bspline = true;		
	} else if (strcmp(argv[5], "bspline2") == 0) { // DONT USE, TESTING ONLY
		RegisterImages_MSBSpline(fixed_image, moving_image, finalTransformBSpline);
		resample->SetTransform(finalTransformBSpline);
	} else if (strcmp(argv[5], "affine") == 0) {
		RegisterImages_MSAffine(fixed_image, moving_image, finalTransformAffine);		
		SmoothAndNormalize(fixed_image, fixed_image);
		SmoothAndNormalize(moving_image, moving_image);
		RegisterImages_MIAffine(fixed_image, moving_image, finalTransformAffine);
		resample->SetTransform(finalTransformAffine);
	} else if (strcmp(argv[5], "affine2") == 0) { // DONT USE, TESTING ONLY
		SmoothAndNormalize(fixed_image, fixed_image);
		SmoothAndNormalize(moving_image, moving_image);
		RegisterImages_MIAffine(fixed_image, moving_image, finalTransformAffine);
		resample->SetTransform(finalTransformAffine);
	} else {
		cerr << "BAD INPUT" << endl;
		exit(EXIT_FAILURE);
	}
	clock.Stop();
	
	std::cout << " Time Elapsed: " << clock.GetTotal() << std::endl;

	resample->SetInput(original_moving_image);
	resample->SetSize(fixed_image->GetLargestPossibleRegion().GetSize());
	resample->SetOutputOrigin(fixed_image->GetOrigin());
	resample->SetOutputSpacing(fixed_image->GetSpacing());
	resample->SetOutputDirection(fixed_image->GetDirection());
	resample->Update();

	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(argv[3]);
	writer->SetInput(resample->GetOutput());


	itk::TransformFileWriter::Pointer t_writer = itk::TransformFileWriter::New();

	try {
		writer->Update();
	} catch (itk::ExceptionObject & err) {
		std::cout << "ExceptionObject caught !" << std::endl;
		std::cout << err << std::endl;
		exit(EXIT_FAILURE);
	}

	try {		
		t_writer->SetFileName(argv[4]);
		t_writer->SetInput(resample->GetTransform());
		t_writer->Update();
	} catch (itk::ExceptionObject & err) {
		std::cout << "ExceptionObject caught !" << std::endl;
		std::cout << err << std::endl;
		exit(EXIT_FAILURE);
	}


	return EXIT_SUCCESS;
}


void SmoothAndNormalize(ImageType::Pointer & input_image, ImageType::Pointer & output_image, double variance) {

	// NORMALIZE BEGIN
	typedef itk::NormalizeImageFilter<ImageType, ImageType> NormalizeFilterType;
	NormalizeFilterType::Pointer normalizer = NormalizeFilterType::New();
	normalizer->SetInput(input_image);
	normalizer->Update();	
	// NORMALIZE END


	// BLUR BEGIN
	typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> GaussianFilterType;
	GaussianFilterType::Pointer smoother = GaussianFilterType::New();
	smoother->SetVariance(variance);
	smoother->SetInput(normalizer->GetOutput());
	smoother->Update();
	output_image = smoother->GetOutput();
	// BLUR END
}

void RegisterImages_MSAffine(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image, TransformTypeAffine::Pointer & out_transform) {
	typedef itk::MultiResolutionImageRegistrationMethod<ImageType, ImageType> MultiResRegistrationType;
	typedef itk::MultiResolutionPyramidImageFilter<ImageType, ImageType> ImagePyramidType;
	typedef itk::LinearInterpolateImageFunction<ImageType, double> InterpolatorType;
	typedef itk::ImageRegistrationMethod<ImageType, ImageType> RegistrationType;
	typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
	//typedef itk::MutualInformationImageToImageMetric<ImageType, ImageType> MetricTypeMI;
	//typedef itk::MattesMutualInformationImageToImageMetric< ImageType, ImageType > MetricTypeMI;
	typedef itk::MeanSquaresImageToImageMetric<ImageType, ImageType> MetricType;
	typedef RegistrationType::ParametersType ParametersType;

	MetricType::Pointer metric = MetricType::New();
	TransformTypeAffine::Pointer transform = TransformTypeAffine::New();
	OptimizerType::Pointer optimizer = OptimizerType::New();
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	MultiResRegistrationType::Pointer registration = MultiResRegistrationType::New();

	ImagePyramidType::Pointer fixedImagePyramid = ImagePyramidType::New();
	ImagePyramidType::Pointer movingImagePyramid = ImagePyramidType::New();

	// REGISTRATION PARAMETERS BEGIN
	unsigned int n_levels = 2;
	unsigned int starting_sfactor = 2;
	fixedImagePyramid->SetNumberOfLevels(n_levels);
	fixedImagePyramid->SetStartingShrinkFactors(starting_sfactor);
	movingImagePyramid->SetNumberOfLevels(n_levels);
	movingImagePyramid->SetStartingShrinkFactors(starting_sfactor);
	registration->SetNumberOfThreads(1);
	// REGISTRATION PARAMETERS END

	// COPY INPUT IMAGES
	ImageType::Pointer fixedImage = ImageType::New();
	ImageType::Pointer movingImage = ImageType::New();

	//DuplicatorType::Pointer duplicator = DuplicatorType::New();
	//duplicator->SetInputImage(fixed_image);
	//duplicator->Update();
	//fixedImage = duplicator->GetOutput();

	//duplicator->SetInputImage(moving_image);
	//duplicator->Update();
	//movingImage = duplicator->GetOutput();
	//


	// EXTRACT BRAINS

	getBrain(fixed_image, fixedImage);
	getBrain(moving_image, movingImage);

	// 

	// OPTIMIZER PARAMETERS BEGIN
	optimizer->SetMaximumStepLength(1.0);
	optimizer->SetMinimumStepLength(0.01);
	optimizer->SetNumberOfIterations(25);

	//optimizer->MaximizeOn();
	optimizer->MinimizeOn();

	double translationScale = 1.0 / 1000.0;
	typedef OptimizerType::ScalesType OptimizerScalesType;
	OptimizerScalesType optimizerScales(transform->GetNumberOfParameters());
	optimizerScales[0] = 1.0;
	optimizerScales[1] = 1.0;
	optimizerScales[2] = 1.0;
	optimizerScales[3] = 1.0;
	optimizerScales[4] = 1.0;
	optimizerScales[5] = 1.0;
	optimizerScales[6] = 1.0;
	optimizerScales[7] = 1.0;
	optimizerScales[8] = 1.0;
	optimizerScales[9] = translationScale;
	optimizerScales[10] = translationScale;
	optimizerScales[11] = translationScale;
	optimizer->SetScales(optimizerScales);

	double fixed_volume;
	vector<double> fixed_center;
	getVolumeAndCenter(fixedImage, fixed_volume, fixed_center);

	double moving_volume;
	vector<double> moving_center;
	getVolumeAndCenter(movingImage, moving_volume, moving_center);

	double scale = pow(fixed_volume / moving_volume, 1.0 / 3.0);
	vector<double> transl(3, 0);
	transl[0] = (fixed_center[0] - moving_center[0]) * scale;
	transl[1] = (fixed_center[1] - moving_center[1]) * scale;
	transl[2] = (fixed_center[2] - moving_center[2]) * scale;

	ParametersType initialParameters(transform->GetNumberOfParameters());
	initialParameters[0] = scale;
	initialParameters[1] = 0.0;
	initialParameters[2] = 0.0;
	initialParameters[3] = 0.0;
	initialParameters[4] = scale;
	initialParameters[5] = 0.0;
	initialParameters[6] = 0.0;
	initialParameters[7] = 0.0;
	initialParameters[8] = scale;

	initialParameters[9] = transl[0];
	initialParameters[10] = transl[1];
	initialParameters[11] = -transl[2];

	registration->SetInitialTransformParameters(initialParameters);
	// OPTIMIZER PARAMETERS END

	// OBSERVER
	CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
	optimizer->AddObserver(itk::IterationEvent(), observer);
	

	RegistrationInterfaceCommand<MultiResRegistrationType>::Pointer command = RegistrationInterfaceCommand<MultiResRegistrationType>::New();
	registration->AddObserver(itk::IterationEvent(), command);
	//

	registration->SetMetric(metric);
	registration->SetOptimizer(optimizer);
	registration->SetTransform(transform);
	registration->SetInterpolator(interpolator);

	registration->SetFixedImagePyramid(fixedImagePyramid);
	registration->SetMovingImagePyramid(movingImagePyramid);

	registration->SetFixedImage(fixedImage);
	registration->SetMovingImage(movingImage);
	registration->SetFixedImageRegion(fixedImage->GetBufferedRegion());

	registration->SetNumberOfLevels(n_levels);
	//std::cout << " Numb. Samples = " << metric->GetNumberOfSpatialSamples() << std::endl;

	try {
		std::cout << "-= Mean Square Error Affine Transform Image Registration =-" << std::endl;
		cout << "Levels: " << fixedImagePyramid->GetNumberOfLevels() << endl;
		cout << "Schedule: " << endl << fixedImagePyramid->GetSchedule() << endl;
		registration->Update();
		std::cout << "Optimizer stop condition: " << registration->GetOptimizer()->GetStopConditionDescription() << std::endl;
	} catch (itk::ExceptionObject & err) {
		std::cout << "ExceptionObject caught !" << std::endl;
		std::cout << err << std::endl;
		exit(EXIT_FAILURE);
	}

	ParametersType finalParameters = registration->GetLastTransformParameters();

	std::cout << "Final Parameters: " << finalParameters << std::endl;

	unsigned int numberOfIterations = optimizer->GetCurrentIteration();

	double bestValue = optimizer->GetValue();

	// Print out results
	std::cout << std::endl;
	std::cout << "Result = " << std::endl;
	std::cout << " Iterations    = " << numberOfIterations << std::endl;
	std::cout << " Metric value  = " << bestValue << std::endl;
	std::cout << " Numb. Samples = " << metric->GetNumberOfSpatialSamples() << std::endl;
	cout << endl;

	out_transform = TransformTypeAffine::New();
	out_transform->SetParameters(finalParameters);
	out_transform->SetFixedParameters(transform->GetFixedParameters());
}

void RegisterImages_MIAffine(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image, TransformTypeAffine::Pointer & out_transform) {
	typedef itk::MultiResolutionImageRegistrationMethod<ImageType, ImageType> MultiResRegistrationType;
	typedef itk::MultiResolutionPyramidImageFilter<ImageType, ImageType> ImagePyramidType;
	typedef itk::LinearInterpolateImageFunction<ImageType, double> InterpolatorType;
	typedef itk::ImageRegistrationMethod<ImageType, ImageType> RegistrationType;
	typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
	//typedef itk::MutualInformationImageToImageMetric<ImageType, ImageType> MetricTypeMI;
	typedef itk::MattesMutualInformationImageToImageMetric< ImageType, ImageType > MetricTypeMI;
	typedef RegistrationType::ParametersType ParametersType;

	MetricTypeMI::Pointer metric = MetricTypeMI::New();
	TransformTypeAffine::Pointer transform = TransformTypeAffine::New();
	OptimizerType::Pointer optimizer = OptimizerType::New();
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	MultiResRegistrationType::Pointer registration = MultiResRegistrationType::New();

	ImagePyramidType::Pointer fixedImagePyramid = ImagePyramidType::New();
	ImagePyramidType::Pointer movingImagePyramid = ImagePyramidType::New();

	// REGISTRATION PARAMETERS BEGIN
	unsigned int n_levels = 2;
	unsigned int starting_sfactor = 2;
	fixedImagePyramid->SetNumberOfLevels(n_levels);
	fixedImagePyramid->SetStartingShrinkFactors(starting_sfactor);
	movingImagePyramid->SetNumberOfLevels(n_levels);
	movingImagePyramid->SetStartingShrinkFactors(starting_sfactor);
	registration->SetNumberOfThreads(1);
	// REGISTRATION PARAMETERS END



	// OPTIMIZER PARAMETERS BEGIN
	optimizer->SetMaximumStepLength(0.1);
	optimizer->SetMinimumStepLength(0.001);
	optimizer->SetNumberOfIterations(300);
	//optimizer->SetGradientMagnitudeTolerance(1.0);

	//optimizer->MaximizeOn();
	optimizer->MinimizeOn();

	double translationScale = 1.0 / 1000.0;
	typedef OptimizerType::ScalesType OptimizerScalesType;
	OptimizerScalesType optimizerScales(transform->GetNumberOfParameters());
	optimizerScales[0] = 1.0;
	optimizerScales[1] = 1.0;
	optimizerScales[2] = 1.0;
	optimizerScales[3] = 1.0;
	optimizerScales[4] = 1.0;
	optimizerScales[5] = 1.0;
	optimizerScales[6] = 1.0;
	optimizerScales[7] = 1.0;
	optimizerScales[8] = 1.0;
	optimizerScales[9] = translationScale;
	optimizerScales[10] = translationScale;
	optimizerScales[11] = translationScale;
	optimizer->SetScales(optimizerScales);

	registration->SetInitialTransformParameters(out_transform->GetParameters());
	// OPTIMIZER PARAMETERS END

	// METRIC PARAMETERS BEGIN
	metric->SetNumberOfHistogramBins(50);
	const unsigned int numberOfPixels = fixed_image->GetBufferedRegion().GetNumberOfPixels();
	const unsigned int numberOfSamples = static_cast<unsigned int>(numberOfPixels * 0.3);
	metric->SetNumberOfSpatialSamples(numberOfSamples);
	// METRIC PARAMETERS END

	// OBSERVER
	CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
	optimizer->AddObserver(itk::IterationEvent(), observer);

	RegistrationInterfaceCommand<MultiResRegistrationType>::Pointer command = RegistrationInterfaceCommand<MultiResRegistrationType>::New();
	registration->AddObserver(itk::IterationEvent(), command);
	//

	registration->SetMetric(metric);
	registration->SetOptimizer(optimizer);
	registration->SetTransform(transform);
	registration->SetInterpolator(interpolator);

	registration->SetFixedImagePyramid(fixedImagePyramid);
	registration->SetMovingImagePyramid(movingImagePyramid);

	registration->SetFixedImage(fixed_image);
	registration->SetMovingImage(moving_image);
	registration->SetFixedImageRegion(fixed_image->GetBufferedRegion());

	registration->SetNumberOfLevels(n_levels);
	//std::cout << " Numb. Samples = " << metric->GetNumberOfSpatialSamples() << std::endl;

	try {
		std::cout << "-= Mutual Information Affine Transform Image Registration =-" << std::endl;
		cout << "Levels: " << fixedImagePyramid->GetNumberOfLevels() << endl;
		cout << "Schedule: " << endl << fixedImagePyramid->GetSchedule() << endl;
		registration->Update();
		std::cout << "Optimizer stop condition: " << registration->GetOptimizer()->GetStopConditionDescription() << std::endl;
	} catch (itk::ExceptionObject & err) {
		std::cout << "ExceptionObject caught !" << std::endl;
		std::cout << err << std::endl;
		exit(EXIT_FAILURE);
	}

	ParametersType finalParameters = registration->GetLastTransformParameters();

	std::cout << "Final Parameters: " << finalParameters << std::endl;

	unsigned int numberOfIterations = optimizer->GetCurrentIteration();

	double bestValue = optimizer->GetValue();

	// Print out results
	std::cout << std::endl;
	std::cout << "Result = " << std::endl;
	std::cout << " Iterations    = " << numberOfIterations << std::endl;
	std::cout << " Metric value  = " << bestValue << std::endl;
	std::cout << " Numb. Samples = " << metric->GetNumberOfSpatialSamples() << std::endl;
	cout << endl;

	out_transform = TransformTypeAffine::New();
	out_transform->SetParameters(finalParameters);
	out_transform->SetFixedParameters(transform->GetFixedParameters());
}

void RegisterImages_MIAffineMulti(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image, TransformTypeAffine::Pointer & out_transform) {
	typedef itk::MultiResolutionImageRegistrationMethod<ImageType, ImageType> MultiResRegistrationType;
	typedef itk::MultiResolutionPyramidImageFilter<ImageType, ImageType> ImagePyramidType;
	typedef itk::LinearInterpolateImageFunction<ImageType, double> InterpolatorType;
	typedef itk::ImageRegistrationMethod<ImageType, ImageType> RegistrationType;
	typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
	//typedef itk::MutualInformationImageToImageMetric<ImageType, ImageType> MetricTypeMI;
	typedef itk::MattesMutualInformationImageToImageMetric< ImageType, ImageType > MetricTypeMI;
	typedef RegistrationType::ParametersType ParametersType;

	MetricTypeMI::Pointer metric = MetricTypeMI::New();
	TransformTypeAffine::Pointer transform = TransformTypeAffine::New();
	OptimizerType::Pointer optimizer = OptimizerType::New();
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	MultiResRegistrationType::Pointer registration = MultiResRegistrationType::New();

	ImagePyramidType::Pointer fixedImagePyramid = ImagePyramidType::New();
	ImagePyramidType::Pointer movingImagePyramid = ImagePyramidType::New();

	// REGISTRATION PARAMETERS BEGIN
	unsigned int n_levels = 2;
	unsigned int starting_sfactor = 2;
	fixedImagePyramid->SetNumberOfLevels(n_levels);
	fixedImagePyramid->SetStartingShrinkFactors(starting_sfactor);
	movingImagePyramid->SetNumberOfLevels(n_levels);
	movingImagePyramid->SetStartingShrinkFactors(starting_sfactor);
	registration->SetNumberOfThreads(1);
	// REGISTRATION PARAMETERS END



	// OPTIMIZER PARAMETERS BEGIN
	optimizer->SetMaximumStepLength(1.0);
	optimizer->SetMinimumStepLength(0.001);
	optimizer->SetNumberOfIterations(300);
	
	//optimizer->MaximizeOn();
	optimizer->MinimizeOn();

	double translationScale = 1.0 / 1000.0;
	typedef OptimizerType::ScalesType OptimizerScalesType;
	OptimizerScalesType optimizerScales(transform->GetNumberOfParameters());
	optimizerScales[0] = 1.0;
	optimizerScales[1] = 1.0;
	optimizerScales[2] = 1.0;
	optimizerScales[3] = 1.0;
	optimizerScales[4] = 1.0;
	optimizerScales[5] = 1.0;
	optimizerScales[6] = 1.0;
	optimizerScales[7] = 1.0;
	optimizerScales[8] = 1.0;
	optimizerScales[9] = translationScale;
	optimizerScales[10] = translationScale;
	optimizerScales[11] = translationScale;
	optimizer->SetScales(optimizerScales);

	double fixed_volume;
	vector<double> fixed_center;
	getVolumeAndCenter(fixed_image, fixed_volume, fixed_center);

	double moving_volume;
	vector<double> moving_center;
	getVolumeAndCenter(moving_image, moving_volume, moving_center);

	double scale = pow(fixed_volume / moving_volume, 1.0 / 3.0);
	vector<double> transl(3, 0);
	transl[0] = (fixed_center[0] - moving_center[0]) * scale;
	transl[1] = (fixed_center[1] - moving_center[1]) * scale;
	transl[2] = (fixed_center[2] - moving_center[2]) * scale;

	ParametersType initialParameters(transform->GetNumberOfParameters());
	initialParameters[0] = scale;
	initialParameters[1] = 0.0;
	initialParameters[2] = 0.0;
	initialParameters[3] = 0.0;
	initialParameters[4] = scale;
	initialParameters[5] = 0.0;
	initialParameters[6] = 0.0;
	initialParameters[7] = 0.0;
	initialParameters[8] = scale;

	initialParameters[9] = transl[0];
	initialParameters[10] = transl[1];
	initialParameters[11] = -transl[2];

	registration->SetInitialTransformParameters(initialParameters);
	// OPTIMIZER PARAMETERS END

	// METRIC PARAMETERS BEGIN
	metric->SetNumberOfHistogramBins(500);
	const unsigned int numberOfPixels = fixed_image->GetBufferedRegion().GetNumberOfPixels();
	const unsigned int numberOfSamples = static_cast<unsigned int>(numberOfPixels * 0.1);
	metric->SetNumberOfSpatialSamples(numberOfSamples);
	// METRIC PARAMETERS END

	// OBSERVER
	CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
	optimizer->AddObserver(itk::IterationEvent(), observer);

	RegistrationInterfaceCommand<MultiResRegistrationType>::Pointer command = RegistrationInterfaceCommand<MultiResRegistrationType>::New();
	registration->AddObserver(itk::IterationEvent(), command);
	//

	registration->SetMetric(metric);
	registration->SetOptimizer(optimizer);
	registration->SetTransform(transform);
	registration->SetInterpolator(interpolator);	

	registration->SetFixedImagePyramid(fixedImagePyramid);
	registration->SetMovingImagePyramid(movingImagePyramid);

	registration->SetFixedImage(fixed_image);
	registration->SetMovingImage(moving_image);
	registration->SetFixedImageRegion(fixed_image->GetBufferedRegion());

	registration->SetNumberOfLevels(n_levels);
	//std::cout << " Numb. Samples = " << metric->GetNumberOfSpatialSamples() << std::endl;

	try {
		std::cout << "-= Mutual Information Affine Transform Image Registration =-" << std::endl;
		cout << "Levels: " << fixedImagePyramid->GetNumberOfLevels() << endl;
		cout << "Schedule: " << endl << fixedImagePyramid->GetSchedule() << endl;
		registration->Update();
		std::cout << "Optimizer stop condition: " << registration->GetOptimizer()->GetStopConditionDescription() << std::endl;
	} catch (itk::ExceptionObject & err) {
		std::cout << "ExceptionObject caught !" << std::endl;
		std::cout << err << std::endl;
		exit(EXIT_FAILURE);
	}

	ParametersType finalParameters = registration->GetLastTransformParameters();

	std::cout << "Final Parameters: " << finalParameters << std::endl;

	unsigned int numberOfIterations = optimizer->GetCurrentIteration();

	double bestValue = optimizer->GetValue();

	// Print out results
	std::cout << std::endl;
	std::cout << "Result = " << std::endl;
	std::cout << " Iterations    = " << numberOfIterations << std::endl;
	std::cout << " Metric value  = " << bestValue << std::endl;
	std::cout << " Numb. Samples = " << metric->GetNumberOfSpatialSamples() << std::endl;


	out_transform = TransformTypeAffine::New();
	out_transform->SetParameters(finalParameters);
	out_transform->SetFixedParameters(transform->GetFixedParameters());
}

void RegisterImages_MIBSpline(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image, TransformTypeBSpline::Pointer & out_transform) {
	typedef itk::LinearInterpolateImageFunction<ImageType, double> InterpolatorType;
	typedef itk::ImageRegistrationMethod<ImageType, ImageType> RegistrationType;
	typedef itk::LBFGSBOptimizer OptimizerType;
	typedef itk::MattesMutualInformationImageToImageMetric<ImageType, ImageType> MetricTypeMI;
	typedef RegistrationType::ParametersType ParametersType;	


	RegistrationType::Pointer registration = RegistrationType::New();
	TransformTypeBSpline::Pointer transform = TransformTypeBSpline::New();
	MetricTypeMI::Pointer metric = MetricTypeMI::New();
	OptimizerType::Pointer optimizer = OptimizerType::New();
	InterpolatorType::Pointer interpolator = InterpolatorType::New();


	// TRANSFORM PARAMETERS BEGIN
	unsigned int numberOfGridNodesInOneDimension = 25;
	TransformTypeBSpline::PhysicalDimensionsType fixedPhysicalDimensions;
	TransformTypeBSpline::MeshSizeType meshSize;
	TransformTypeBSpline::OriginType fixedOrigin;
	for (unsigned int i = 0; i< 3; i++) {
		fixedOrigin[i] = fixed_image->GetOrigin()[i];
		fixedPhysicalDimensions[i] = fixed_image->GetSpacing()[i] * static_cast<double>(fixed_image->GetLargestPossibleRegion().GetSize()[i] - 1);
	}
	meshSize.Fill(numberOfGridNodesInOneDimension - spline_order);
	transform->SetTransformDomainOrigin(fixedOrigin);
	transform->SetTransformDomainPhysicalDimensions(fixedPhysicalDimensions);
	transform->SetTransformDomainMeshSize(meshSize);
	transform->SetTransformDomainDirection(fixed_image->GetDirection());
	typedef TransformTypeBSpline::ParametersType ParametersType;
	const unsigned int numberOfParameters = transform->GetNumberOfParameters();
	ParametersType parameters(numberOfParameters);
	parameters.Fill(0.0);
	transform->SetParameters(parameters);
	// TRANSFORM PARAMETERS END

	// METRIC PARAMETERS BEGIN
	metric->SetNumberOfHistogramBins(50);
	const unsigned int numberOfPixels = fixed_image->GetBufferedRegion().GetNumberOfPixels();
	const unsigned int numberOfSamples = static_cast<unsigned int>(numberOfPixels * 0.3);
	metric->SetNumberOfSpatialSamples(numberOfSamples);
	// METRIC PARAMETERS END

	// OPTIMIZER PARAMETERS BEGIN
	const unsigned int numParameters = transform->GetNumberOfParameters();
	OptimizerType::BoundSelectionType boundSelect(numParameters);
	OptimizerType::BoundValueType upperBound(numParameters);
	OptimizerType::BoundValueType lowerBound(numParameters);

	boundSelect.Fill(0);
	upperBound.Fill(0.0);
	lowerBound.Fill(0.0);

	optimizer->SetBoundSelection(boundSelect);
	optimizer->SetUpperBound(upperBound);
	optimizer->SetLowerBound(lowerBound);

	optimizer->SetCostFunctionConvergenceFactor(1.e7);
	optimizer->SetProjectedGradientTolerance(5e-6);
	optimizer->SetMaximumNumberOfIterations(500);
	optimizer->SetMaximumNumberOfEvaluations(500);
	optimizer->SetMaximumNumberOfCorrections(100);

	//optimizer->TraceOn();
	//optimizer->MaximizeOn();

	// OPTIMIZER PARAMETERS END

	// OBSERVER
	CommandIterationUpdateBSpline::Pointer observer = CommandIterationUpdateBSpline::New();
	optimizer->AddObserver(itk::IterationEvent(), observer);
	//

	registration->SetMetric(metric);
	registration->SetOptimizer(optimizer);
	registration->SetTransform(transform);
	registration->SetInterpolator(interpolator);

	registration->SetInitialTransformParameters(transform->GetParameters());	

	registration->SetFixedImage(fixed_image);
	registration->SetMovingImage(moving_image);
	registration->SetFixedImageRegion(fixed_image->GetBufferedRegion());
	registration->SetNumberOfThreads(2);


	try {
		std::cout << "-= Mutual Information BSpline Transform Image Registration =-" << std::endl;
		registration->Update();
		std::cout << "Optimizer stop condition: " << registration->GetOptimizer()->GetStopConditionDescription() << std::endl;
	} catch (itk::ExceptionObject & err) {
		std::cout << "ExceptionObject caught !" << std::endl;
		std::cout << err << std::endl;
		exit(EXIT_FAILURE);
	}

	ParametersType finalParameters = registration->GetLastTransformParameters();

	//std::cout << "Final Parameters: " << finalParameters << std::endl;	

	double bestValue = optimizer->GetValue();

	// Print out results
	std::cout << std::endl;
	std::cout << "Result = " << std::endl;
	std::cout << " Metric value  = " << bestValue << std::endl;
	std::cout << " Numb. Samples = " << metric->GetNumberOfSpatialSamples() << std::endl;

	out_transform = TransformTypeBSpline::New();
	out_transform->SetTransformDomainOrigin(fixedOrigin);
	out_transform->SetTransformDomainPhysicalDimensions(fixedPhysicalDimensions);
	out_transform->SetTransformDomainMeshSize(meshSize);
	out_transform->SetTransformDomainDirection(fixed_image->GetDirection());
	out_transform->SetParameters(finalParameters);
}


void RegisterImages_MSBSpline(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image, TransformTypeBSpline::Pointer & out_transform) {
	
	typedef itk::LBFGSOptimizer OptimizerType;
	typedef itk::MeanSquaresImageToImageMetric<ImageType, ImageType> MetricType;
	typedef itk::LinearInterpolateImageFunction<ImageType, double>    InterpolatorType;
	typedef itk::ImageRegistrationMethod<ImageType, ImageType> RegistrationType;

	MetricType::Pointer         metric = MetricType::New();
	OptimizerType::Pointer      optimizer = OptimizerType::New();
	InterpolatorType::Pointer   interpolator = InterpolatorType::New();
	RegistrationType::Pointer   registration = RegistrationType::New();


// COPY INPUT IMAGES
	ImageType::Pointer fixedImage = ImageType::New();
	ImageType::Pointer movingImage = ImageType::New();
//


// EXTRACT BRAINS
	getBrain(fixed_image, fixedImage);
	getBrain(moving_image, movingImage);

	//getLabelImage(fixed_image, fixedImage, k);
	//getLabelImage(moving_image, movingImage, k);

// 

// REGISTRATION PARAMETERS
	registration->SetNumberOfThreads(1);
	registration->SetMetric(metric);
	registration->SetOptimizer(optimizer);
	registration->SetInterpolator(interpolator);

	TransformTypeBSpline::Pointer  transform = TransformTypeBSpline::New();
	registration->SetTransform(transform);

	registration->SetFixedImage(fixedImage);
	registration->SetMovingImage(movingImage);

	ImageType::RegionType fixedRegion = fixedImage->GetBufferedRegion();
	registration->SetFixedImageRegion(fixedRegion);


// TRANSFORM PARAMETERS
	TransformTypeBSpline::PhysicalDimensionsType   fixedPhysicalDimensions;
	TransformTypeBSpline::MeshSizeType             meshSize;
	for (unsigned int i = 0; i < 3; i++) {
		fixedPhysicalDimensions[i] = fixedImage->GetSpacing()[i] * static_cast<double>(fixedImage->GetLargestPossibleRegion().GetSize()[i] - 1);
	}
	unsigned int numberOfGridNodesInOneDimension = 9;
	meshSize.Fill(numberOfGridNodesInOneDimension - spline_order);
	transform->SetTransformDomainOrigin(fixedImage->GetOrigin());
	transform->SetTransformDomainPhysicalDimensions(fixedPhysicalDimensions);
	transform->SetTransformDomainMeshSize(meshSize);
	transform->SetTransformDomainDirection(fixedImage->GetDirection());

	typedef TransformTypeBSpline::ParametersType ParametersType;
	const unsigned int numberOfParameters =	transform->GetNumberOfParameters();

	ParametersType parameters(numberOfParameters);
	parameters.Fill(0.0);
	transform->SetParameters(parameters);
	registration->SetInitialTransformParameters(transform->GetParameters());
//

// OPTIMIZER PARAMETERS
	optimizer->SetGradientConvergenceTolerance(0.01);
	optimizer->SetLineSearchAccuracy(0.9);
	optimizer->SetDefaultStepLength(.5);
	optimizer->TraceOn();
	optimizer->SetMaximumNumberOfFunctionEvaluations(5);
//

	std::cout << std::endl << "Starting Registration" << std::endl;

	try {
		std::cout << "-= Mean Square Error BSpline Transform Image Registration =-" << std::endl;
		registration->Update();
		std::cout << "Optimizer stop condition: " << registration->GetOptimizer()->GetStopConditionDescription() << std::endl;
	} catch (itk::ExceptionObject & err) {
		std::cout << "ExceptionObject caught !" << std::endl;
		std::cout << err << std::endl;
		exit(EXIT_FAILURE);
	}

	ParametersType finalParameters = registration->GetLastTransformParameters();
	double bestValue = optimizer->GetValue();

	// Print out results
	std::cout << std::endl;
	std::cout << "Result = " << std::endl;
	std::cout << " Metric value  = " << bestValue << std::endl;
	std::cout << " Numb. Samples = " << metric->GetNumberOfSpatialSamples() << std::endl;

	out_transform = TransformTypeBSpline::New();
	out_transform->SetTransformDomainOrigin(fixedImage->GetOrigin());
	out_transform->SetTransformDomainPhysicalDimensions(fixedPhysicalDimensions);
	out_transform->SetTransformDomainMeshSize(meshSize);
	out_transform->SetTransformDomainDirection(fixed_image->GetDirection());
	out_transform->SetParameters(finalParameters);
}

void getVolumeAndCenter(ImageType::Pointer & image, double & volume, vector<double> & center) {
	itk::ImageRegionIteratorWithIndex<ImageType> imageIterator(image, image->GetBufferedRegion());

	int n_brain_voxels = 0;
	center = vector<double>(3, 0);
	while (!imageIterator.IsAtEnd()) {

		double val = imageIterator.Get();
		if (val > 0) {
			++n_brain_voxels;
			center[0] += imageIterator.GetIndex()[0] * image->GetSpacing()[0];
			center[1] += imageIterator.GetIndex()[1] * image->GetSpacing()[1];
			center[2] += imageIterator.GetIndex()[2] * image->GetSpacing()[2];
		}
		++imageIterator;
	}

	center[0] /= n_brain_voxels;
	center[1] /= n_brain_voxels;
	center[2] /= n_brain_voxels;

	//center[0] *= image->GetSpacing()[0];
	//center[1] *= image->GetSpacing()[1];
	//center[2] *= image->GetSpacing()[2];

	double vox_volume = image->GetSpacing()[0] * image->GetSpacing()[1] * image->GetSpacing()[2];
	volume = vox_volume * double(n_brain_voxels);
}

void changeBasis(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image) {

	typedef itk::IdentityTransform<double, 3> TransformTypeI;
	TransformTypeI::Pointer testtrans = TransformTypeI::New();
	typedef itk::ResampleImageFilter<ImageType, ImageType > ResampleFilterType;
	ResampleFilterType::Pointer resample = ResampleFilterType::New();
	resample->SetTransform(testtrans);
	resample->SetInput(moving_image);
	resample->SetSize(fixed_image->GetLargestPossibleRegion().GetSize());
	resample->SetOutputOrigin(fixed_image->GetOrigin());
	resample->SetOutputSpacing(fixed_image->GetSpacing());
	resample->SetOutputDirection(fixed_image->GetDirection());
	resample->Update();
	moving_image = resample->GetOutput();

}

void getBrain(ImageType::Pointer & image, ImageType::Pointer & out_image, int n_bins) {
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

	DuplicatorType::Pointer duplicator = DuplicatorType::New();
	duplicator->SetInputImage(image);
	duplicator->Update();
	out_image = duplicator->GetOutput();

	itk::ImageRegionIteratorWithIndex<ImageType> imgIterator(out_image, out_image->GetBufferedRegion());
	while (!imgIterator.IsAtEnd()) {

		double val = imgIterator.Get();

		if (lower_bound <= val && val <= upper_bound) {
			imgIterator.Set(0);
		} else {
			imgIterator.Set(1);
		}

		++imgIterator;
	}


	typedef itk::MedianImageFilter<ImageType, ImageType > FilterType;
	FilterType::Pointer medianFilter = FilterType::New();
	FilterType::InputSizeType radius;
	radius.Fill(1);

	medianFilter->SetRadius(radius);
	medianFilter->SetInput(out_image);
	medianFilter->Update();

	out_image = medianFilter->GetOutput();
}


void initTrans(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image) {
	double fixed_volume;
	vector<double> fixed_center;
	getVolumeAndCenter(fixed_image, fixed_volume, fixed_center);
	cout << "Fixed Volume Before: " << fixed_volume << endl;
	cout << "Fixed Center Before: [" << fixed_center[0] << "," << fixed_center[1] << "," << fixed_center[2] << "]" << endl;

	double moving_volume;
	vector<double> moving_center;
	getVolumeAndCenter(moving_image, moving_volume, moving_center);
	cout << "Moving Volume Before: " << moving_volume << endl;
	cout << "Moving Center Before: [" << moving_center[0] << "," << moving_center[1] << "," << moving_center[2] << "]" << endl;


	double scale = pow(fixed_volume / moving_volume, 1.0 / 3.0);
	//double scale = 1.0;
	vector<double> transl(3, 0);
	transl[0] = (fixed_center[0] - moving_center[0]) * scale;
	transl[1] = (fixed_center[1] - moving_center[1]) * scale;
	transl[2] = (fixed_center[2] - moving_center[2]) * scale;

	TransformTypeAffine::Pointer transform = TransformTypeAffine::New();
	typedef TransformTypeAffine::ParametersType ParametersType;

	ParametersType initialParameters(transform->GetNumberOfParameters());
	initialParameters[0] = scale;
	initialParameters[1] = 0.0;
	initialParameters[2] = 0.0;
	initialParameters[3] = 0.0;
	initialParameters[4] = scale;
	initialParameters[5] = 0.0;
	initialParameters[6] = 0.0;
	initialParameters[7] = 0.0;
	initialParameters[8] = scale;

	initialParameters[9] = transl[0];
	initialParameters[10] = transl[1];
	initialParameters[11] = -transl[2];

	//initialParameters[9] = 0.0;
	//initialParameters[10] = 0.0;
	//initialParameters[11] = 0.0;

	transform->SetParameters(initialParameters);

	cout << endl << "transform: " << endl << initialParameters << endl << endl;

	typedef itk::ResampleImageFilter<ImageType, ImageType > ResampleFilterType;
	ResampleFilterType::Pointer resample = ResampleFilterType::New();
	resample->SetTransform(transform);
	resample->SetInput(moving_image);
	resample->SetSize(fixed_image->GetBufferedRegion().GetSize());
	resample->SetOutputOrigin(fixed_image->GetOrigin());
	resample->SetOutputSpacing(fixed_image->GetSpacing());
	resample->SetOutputDirection(fixed_image->GetDirection());
	resample->Update();
	moving_image = resample->GetOutput();

	//double fixed_volume;
	//vector<double> fixed_center;
	getVolumeAndCenter(fixed_image, fixed_volume, fixed_center);
	cout << "Fixed Volume After: " << fixed_volume << endl;
	cout << "Fixed Center After: [" << fixed_center[0] << "," << fixed_center[1] << "," << fixed_center[2] << "]" << endl;

	//double moving_volume;
	//vector<double> moving_center;
	getVolumeAndCenter(moving_image, moving_volume, moving_center);
	cout << "Moving Volume After: " << moving_volume << endl;
	cout << "Moving Center After: [" << moving_center[0] << "," << moving_center[1] << "," << moving_center[2] << "]" << endl;
}

void getLabelImage(ImageType::Pointer & image, ImageType::Pointer & out_image, int n_classes) {

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

	typedef itk::CastImageFilter<Image<unsigned char, 3>, ImageType> CastFilterType;
	CastFilterType::Pointer castFilter = CastFilterType::New();

	castFilter->SetInput(kmeansFilter->GetOutput());
	castFilter->Update();

	out_image = castFilter->GetOutput();
	//KMeansFilterType::ParametersType eMeans = kmeansFilter->GetFinalMeans();

	//for (unsigned int i = 0; i < eMeans.Size(); ++i) {
	//	std::cout << "cluster[" << i << "] ";
	//	std::cout << " estimated mean : " << eMeans[i] << std::endl;
	//}

}

void getInverseTfm(const ImageType::Pointer & fixed_image, const TransformTypeBSpline::Pointer & tfm, TransformTypeDis::Pointer & itfm) {
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
	IDF_filter->SetNumberOfIterations(100);

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