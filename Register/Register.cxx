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

#include <math.h>
#include <iostream>
#include <string>


typedef itk::Image< double, 3 > ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter< ImageType  > WriterType;
typedef itk::ImageDuplicator< ImageType > DuplicatorType;


const unsigned int spline_order = 3;
typedef itk::BSplineTransform<double, 3, spline_order> TransformTypeBSpline;
typedef itk::AffineTransform<double, 3> TransformTypeAffine;


typedef itk::MeanSquaresImageToImageMetric<ImageType, ImageType> MetricTypeMS;



using namespace std;

template <typename TRegistration>
class RegistrationInterfaceCommand : public itk::Command {
public:
	typedef  RegistrationInterfaceCommand   Self;
	typedef  itk::Command                   Superclass;
	typedef  itk::SmartPointer<Self>        Pointer;
	itkNewMacro(Self);

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
			//optimizer->SetMaximumStepLength(10.00);
			optimizer->SetMinimumStepLength(optimizer->GetMinimumStepLength() * pow(10, registration->GetNumberOfLevels()-1));
		} else {
			optimizer->SetMaximumStepLength(optimizer->GetMinimumStepLength() * 2.0);			
			optimizer->SetMinimumStepLength(optimizer->GetMinimumStepLength() * 0.1);
		
			
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

protected:
	CommandIterationUpdate() {};

public:
	typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
	typedef   const OptimizerType *                  OptimizerPointer;

	void Execute(itk::Object *caller, const itk::EventObject & event) {
		Execute((const itk::Object *)caller, event);
	}

	void Execute(const itk::Object * object, const itk::EventObject & event) {
		OptimizerPointer optimizer =
			dynamic_cast< OptimizerPointer >(object);
		if (!itk::IterationEvent().CheckEvent(&event)) {
			return;
		}
		std::cout << "Iteration: " << optimizer->GetCurrentIteration() << "\t";
		std::cout << "Metric Value: " << optimizer->GetValue() << "\t";
		std::cout << "Current Step Length: " << optimizer->GetCurrentStepLength() << endl;
		//std::cout << optimizer->GetCurrentPosition() << std::endl;
	}
};

class CommandIterationUpdateBSpline : public itk::Command {
public:
	typedef  CommandIterationUpdateBSpline   Self;
	typedef  itk::Command             Superclass;
	typedef itk::SmartPointer<Self>   Pointer;
	itkNewMacro(Self);

	unsigned int iteration;

protected:
	CommandIterationUpdateBSpline() { iteration = 0; };

public:
	typedef itk::LBFGSBOptimizer OptimizerType;
	typedef   const OptimizerType *                  OptimizerPointer;

	void Execute(itk::Object *caller, const itk::EventObject & event) {
		Execute((const itk::Object *)caller, event);
	}

	void Execute(const itk::Object * object, const itk::EventObject & event) {
		OptimizerPointer optimizer =
			dynamic_cast< OptimizerPointer >(object);
		if (!itk::IterationEvent().CheckEvent(&event)) {
			return;
		}
		std::cout << "Iteration: " << iteration << "\t";
		std::cout << "Metric Value: " << optimizer->GetValue() << endl;		
		//std::cout << optimizer->GetCurrentPosition() << std::endl;
		++iteration;
	}
};

void SmoothAndNormalize(ImageType::Pointer & input_image, ImageType::Pointer & output_image, double variance = 2.0);
void RegisterImages_MIAffine(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image, TransformTypeAffine::Pointer & out_transform);
void RegisterImages_MIBSpline(ImageType::Pointer & fixed_image, ImageType::Pointer & moving_image, TransformTypeBSpline::Pointer & out_transform);

int main(int argc, char *argv[])
{

	if (argc < 5) {
		cerr << "INPUT FAIL" << endl;
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

	typedef itk::ImageDuplicator<ImageType> DuplicatorType;
	DuplicatorType::Pointer duplicator = DuplicatorType::New();

	duplicator->SetInputImage(moving_image);
	duplicator->Update();

	ImageType::Pointer original_moving_image = duplicator->GetOutput();
// READ FILES END

	SmoothAndNormalize(fixed_image, fixed_image);
	SmoothAndNormalize(moving_image, moving_image);

	//return 0;
	
	typedef itk::ResampleImageFilter<ImageType, ImageType > ResampleFilterType;
	ResampleFilterType::Pointer resample = ResampleFilterType::New();

	itk::TimeProbe clock;
	clock.Start();
	if (strcmp(argv[4],"bspline") == 0) {
		TransformTypeBSpline::Pointer finalTransform;
		RegisterImages_MIBSpline(fixed_image, moving_image, finalTransform);
		resample->SetTransform(finalTransform);
	} else {
		TransformTypeAffine::Pointer finalTransform;
		RegisterImages_MIAffine(fixed_image, moving_image, finalTransform);
		resample->SetTransform(finalTransform);
	}
	
	clock.Stop();
	std::cout << " Time Elapsed: " << clock.GetTotal() << std::endl;

	resample->SetInput(original_moving_image);
	resample->SetSize(fixed_image->GetLargestPossibleRegion().GetSize());
	resample->SetOutputOrigin(fixed_image->GetOrigin());
	resample->SetOutputSpacing(fixed_image->GetSpacing());
	resample->SetOutputDirection(fixed_image->GetDirection());

	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(argv[3]);
	writer->SetInput(resample->GetOutput());
	writer->Update();

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
	unsigned int n_levels = 3;
	unsigned int starting_sfactor = 7;
	fixedImagePyramid->SetNumberOfLevels(n_levels);
	fixedImagePyramid->SetStartingShrinkFactors(starting_sfactor);
	movingImagePyramid->SetNumberOfLevels(n_levels);
	movingImagePyramid->SetStartingShrinkFactors(starting_sfactor);
	registration->SetNumberOfThreads(4);
	// REGISTRATION PARAMETERS END



	// OPTIMIZER PARAMETERS BEGIN
	optimizer->SetMaximumStepLength(1.0);
	optimizer->SetMinimumStepLength(0.001);
	optimizer->SetNumberOfIterations(300);
	optimizer->MaximizeOn();
	//optimizer->MinimizeOn();

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

	ParametersType initialParameters(transform->GetNumberOfParameters());
	initialParameters[0] = 1.0;
	initialParameters[1] = 0.0;
	initialParameters[2] = 0.0;
	initialParameters[3] = 0.0;
	initialParameters[4] = 1.0;
	initialParameters[5] = 0.0;
	initialParameters[6] = 0.0;
	initialParameters[7] = 0.0;
	initialParameters[8] = 1.0;

	initialParameters[9] = 0.0;
	initialParameters[10] = 0.0;
	initialParameters[11] = 0.0;

	registration->SetInitialTransformParameters(initialParameters);
	// OPTIMIZER PARAMETERS END

	// METRIC PARAMETERS BEGIN
	metric->SetNumberOfHistogramBins(100);
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
	unsigned int numberOfGridNodesInOneDimension = 7;
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
	metric->SetNumberOfHistogramBins(100);
	const unsigned int numberOfPixels = fixed_image->GetBufferedRegion().GetNumberOfPixels();
	const unsigned int numberOfSamples = static_cast<unsigned int>(numberOfPixels * 0.1);
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
	optimizer->SetProjectedGradientTolerance(1e-6);
	optimizer->SetMaximumNumberOfIterations(200);
	optimizer->SetMaximumNumberOfEvaluations(100);
	optimizer->SetMaximumNumberOfCorrections(5);
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
	registration->SetNumberOfThreads(1);


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



