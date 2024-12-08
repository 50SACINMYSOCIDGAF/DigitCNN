{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}

module Main where

import qualified Data.Vector.Unboxed as VU
import qualified Data.Vector as V
import System.Random
import Control.Monad (foldM, when, forever)
import Data.List (foldl')
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy.Char8 as BSL
import qualified Data.Binary as Binary
import Data.Binary.Get
import System.IO
import qualified Data.Aeson as Aeson
import Data.Aeson (ToJSON(..), FromJSON(..), object, (.=))
import GHC.Generics
import Control.DeepSeq
import System.Directory (doesFileExist)
import Text.Printf
import Control.Concurrent
import System.Environment (getArgs)
import Data.String (fromString)
import qualified Data.Vector.Generic as VG
import System.Directory (doesFileExist, getCurrentDirectory)
import System.IO.Unsafe (unsafePerformIO)
import Data.Function (fix)
import Control.Monad (when)


type Weight = Double
type Bias = Double
type Activation = Double
type Delta = Double
type LearningRate = Double

data Neuron = Neuron {
    weights :: !(VU.Vector Weight),
    bias :: !Bias
} deriving (Show, Generic)

instance NFData Neuron

data Layer = Layer {
    neurons :: !(V.Vector Neuron),
    activations :: !(VU.Vector Activation)
} deriving (Show, Generic)

instance NFData Layer

data ConvFilter = ConvFilter {
    filterWeights :: !(V.Vector (VU.Vector Weight)),
    filterBias :: !Bias
} deriving (Show, Generic)

instance NFData ConvFilter

data Dimensions = Dimensions {
    inputSize :: !Int,
    conv1OutputSize :: !Int,
    pool1OutputSize :: !Int,
    conv2OutputSize :: !Int,
    pool2OutputSize :: !Int,
    fcLayerInputSize :: !Int  -- renamed from fcInputSize to avoid naming conflict
} deriving (Show)

calculateDimensions :: Int -> Int -> Int -> Dimensions
calculateDimensions inputWidth filterSize poolSize = 
    Dimensions {
        inputSize = inputWidth * inputWidth,
        conv1OutputSize = conv1Size * conv1Size * 6,  -- 6 filters
        pool1OutputSize = pool1Size * pool1Size * 6,
        conv2OutputSize = conv2Size * conv2Size * 16, -- 16 filters
        pool2OutputSize = pool2Size * pool2Size * 16,
        fcLayerInputSize = pool2Size * pool2Size * 16
    }
    where
        conv1Size = inputWidth - filterSize + 1
        pool1Size = conv1Size `div` poolSize
        conv2Size = pool1Size - filterSize + 1
        pool2Size = conv2Size `div` poolSize

data CNN = CNN {
    convLayer1 :: !(V.Vector ConvFilter),
    convLayer2 :: !(V.Vector ConvFilter),
    fullyConnected :: !Layer,
    outputLayer :: !Layer
} deriving (Show, Generic)

instance NFData CNN

-- Updated these instances with more precise serialization
instance Binary.Binary Neuron where
    put (Neuron w b) = do
        Binary.put (VU.length w :: Int)
        Binary.put (VU.toList w)
        Binary.put b
    get = do
        len <- Binary.get :: Get Int
        ws <- VU.fromList <$> Binary.get
        b <- Binary.get
        return $! Neuron ws b

instance Binary.Binary Layer where
    put (Layer n a) = do
        Binary.put (V.length n :: Int)
        Binary.put (V.toList n)
        Binary.put (VU.length a :: Int)
        Binary.put (VU.toList a)
    get = do
        nLen <- Binary.get :: Get Int
        ns <- V.fromList <$> Binary.get
        aLen <- Binary.get :: Get Int
        as <- VU.fromList <$> Binary.get
        return $! Layer ns as

instance Binary.Binary ConvFilter where
    put (ConvFilter w b) = do
        Binary.put (V.length w :: Int)
        Binary.put (map VU.toList $ V.toList w)
        Binary.put b
    get = do
        wLen <- Binary.get :: Get Int
        ws <- V.fromList . map VU.fromList <$> Binary.get
        b <- Binary.get
        return $! ConvFilter ws b

instance Binary.Binary CNN where
    put (CNN c1 c2 fc out) = do
        Binary.put (V.length c1 :: Int)
        Binary.put (V.toList c1)
        Binary.put (V.length c2 :: Int)
        Binary.put (V.toList c2)
        Binary.put fc
        Binary.put out
    get = do
        c1Len <- Binary.get :: Get Int
        c1 <- V.fromList <$> Binary.get
        c2Len <- Binary.get :: Get Int
        c2 <- V.fromList <$> Binary.get
        fc <- Binary.get
        out <- Binary.get
        return $! CNN c1 c2 fc out

data MNISTData = MNISTData {
    images :: !(V.Vector (VU.Vector Double)),
    labels :: !(V.Vector (VU.Vector Double))
}

sigmoid :: Double -> Double
sigmoid !x 
    | x > 100 = 1.0  -- Prevent exp overflow
    | x < -100 = 0.0
    | isNaN x = 0.5  -- Provide safe default
    | otherwise = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}

sigmoidDerivative :: Double -> Double
sigmoidDerivative !x = s * (1 - s)
  where !s = sigmoid x
{-# INLINE sigmoidDerivative #-}

-- Helper function to calculate convolution
calculateConvolutionOutput :: Int -> Int -> Int
calculateConvolutionOutput inputSize filterSize = inputSize - filterSize + 1

-- Fixed convolve function with proper index handling
convolve :: VU.Vector Double -> ConvFilter -> VU.Vector Double
convolve !input !filter = VU.generate outputSize $ \idx -> unsafePerformIO $ do
    let !outY = idx `div` outputWidth
        !outX = idx `mod` outputWidth
        !result = sum [sum [ ((input VU.! (((outY + fy) * inputWidth) + (outX + fx))) * 
                            (filterWeights filter V.! 0 VU.! (fy * filterSize + fx)))
                          | fx <- [0..filterSize-1]
                          , let inputIdx = ((outY + fy) * inputWidth) + (outX + fx)
                          , inputIdx >= 0 && inputIdx < VU.length input
                          ]
                    | fy <- [0..filterSize-1]
                    , (outY + fy) * inputWidth + outX < VU.length input
                    ] + filterBias filter
        !sigResult = sigmoid result
    
    when (isNaN sigResult || isInfinite sigResult) $
        putStrLn $ "Warning: Invalid result at " ++ show (outX, outY) ++ 
                   " raw: " ++ show result ++ " sigmoid: " ++ show sigResult
    
    return $ if isNaN sigResult || isInfinite sigResult then 0.0 else sigResult
  where
    !inputWidth = 28  -- MNIST image width
    !filterSize = 5   -- 5x5 convolution filter
    !outputWidth = calculateConvolutionOutput inputWidth filterSize
    !outputSize = outputWidth * outputWidth
{-# INLINE convolve #-}

-- Fixed maxPool function with proper index handling
maxPool :: VU.Vector Double -> VU.Vector Double
maxPool !input = VU.generate outputSize $ \idx ->
    let !outY = idx `div` outputWidth
        !outX = idx `mod` outputWidth
        !inY = outY * 2
        !inX = outX * 2
        !indices = [ inY * inputWidth + inX
                  , inY * inputWidth + (inX + 1)
                  , (inY + 1) * inputWidth + inX
                  , (inY + 1) * inputWidth + (inX + 1)
                  ]
        !validVals = [ input VU.! i 
                    | i <- indices
                    , i >= 0 && i < VU.length input
                    ]
    in if null validVals 
       then 0.0 
       else maximum validVals
  where
    !inputWidth = 24   -- Width after first convolution (28 - 5 + 1 = 24)
    !outputWidth = inputWidth `div` 2
    !outputSize = outputWidth * outputWidth
{-# INLINE maxPool #-}

flatten :: VU.Vector Double -> VU.Vector Double
flatten = id
{-# INLINE flatten #-}

layerForward :: Layer -> VU.Vector Double -> Layer
layerForward !layer !input = layer { activations = newActivations }
  where
    !newActivations = VU.generate (V.length $ neurons layer) $ \i ->
        let !neuron = neurons layer V.! i
            !sum = VU.sum $ VU.zipWith (*) (weights neuron) input
        in sigmoid (sum + bias neuron)
{-# INLINE layerForward #-}

randomWeight :: IO Double
randomWeight = randomRIO (-0.05, 0.05) 
{-# INLINE randomWeight #-}

randomConvFilter :: Int -> IO ConvFilter
randomConvFilter !size = do
    !w <- VU.replicateM (size * size) randomWeight
    !b <- randomWeight
    return $! ConvFilter (V.singleton w) b

randomLayer :: Int -> Int -> IO Layer
randomLayer !numNeurons !inputSize = do
    !neurons <- V.replicateM numNeurons $ do
        !ws <- VU.replicateM inputSize randomWeight
        !b <- randomWeight
        return $! Neuron ws b
    return $! Layer neurons (VU.replicate numNeurons 0.0)

-- Fixed initCNN function
initCNN :: IO CNN
initCNN = do
    let dims = calculateDimensions 28 5 2  -- 28x28 input, 5x5 filter, 2x2 pooling
        inputToFC = fcLayerInputSize dims  -- use renamed accessor
    
    !conv1 <- V.replicateM 6 (randomConvFilter 5)  -- 6 5x5 filters
    !conv2 <- V.replicateM 16 (randomConvFilter 5) -- 16 5x5 filters
    !fc <- randomLayer 120 inputToFC              -- Fully connected layer
    !output <- randomLayer 10 120                 -- Output layer (10 digits)
    return $! CNN conv1 conv2 fc output

-- Update forwardProp to handle dimensions properly
forwardProp :: CNN -> VU.Vector Double -> (VU.Vector Double, CNN)
forwardProp !cnn !input =
    let -- First convolution layer
        !conv1Outputs = V.map (convolve input) (convLayer1 cnn)
        !conv1Output = VU.concat $ V.toList conv1Outputs
        
        -- First max pooling
        !pooled1 = maxPool conv1Output
        
        -- Second convolution layer
        !conv2Outputs = V.map (convolve pooled1) (convLayer2 cnn)
        !conv2Output = VU.concat $ V.toList conv2Outputs
        
        -- Second max pooling
        !pooled2 = maxPool conv2Output
        
        -- Flatten and feed through fully connected layers
        !flattened = pooled2  -- flatten is identity since we're already using 1D vectors
        !fcOutput = layerForward (fullyConnected cnn) flattened
        !outputResult = layerForward (outputLayer cnn) (activations fcOutput)
    in (activations outputResult,
        cnn { fullyConnected = fcOutput, outputLayer = outputResult })

backprop :: LearningRate -> CNN -> VU.Vector Double -> VU.Vector Double -> CNN
backprop !lr !cnn !input !target =
    let -- Forward pass
        (!output, !forwardState) = forwardProp cnn input
        
        -- Calculate initial error
        !outputError = VU.zipWith (-) target output
        
        -- Backpropagate through layers
        (!outputGrads, !fcError) = layerBackprop lr (outputLayer forwardState) outputError
        (!fcGrads, !conv2Error) = layerBackprop lr (fullyConnected forwardState) fcError
        !conv2Grads = convLayerBackprop lr (convLayer2 forwardState) conv2Error
        !conv1Grads = convLayerBackprop lr (convLayer1 forwardState) conv2Error
        
        -- Update layers
        !newConv1 = V.zipWith (updateFilter lr) (convLayer1 cnn) conv1Grads
        !newConv2 = V.zipWith (updateFilter lr) (convLayer2 cnn) conv2Grads
        !newFC = updateLayer lr (fullyConnected cnn) fcGrads
        !newOutput = updateLayer lr (outputLayer cnn) outputGrads
        
    in cnn { convLayer1 = newConv1
           , convLayer2 = newConv2
           , fullyConnected = newFC
           , outputLayer = newOutput
           }

-- Fixed layerBackprop function with proper dimension handling
layerBackprop :: LearningRate -> Layer -> VU.Vector Delta -> (V.Vector (VU.Vector Weight, Bias), VU.Vector Delta)
layerBackprop !lr !layer !deltas = 
    let !numNeurons = V.length $ neurons layer
        !numInputs = VU.length $ weights $ V.head $ neurons layer
        
        -- Calculate gradients for each neuron
        !gradients = V.generate numNeurons $ \i ->
            let !neuron = neurons layer V.! i
                !delta = if i < VU.length deltas then deltas VU.! i else 0.0
                !weightGrads = VU.map (* (delta * lr)) (weights neuron)
                !biasGrad = delta * lr
            in (weightGrads, biasGrad)
        
        -- Calculate deltas for next layer
        !nextDeltas = VU.generate numInputs $ \inputIdx ->
            let !contributions = V.map 
                    (\neuron -> 
                        let !neuronIdx = V.length (neurons layer) - 1
                            !delta = if neuronIdx < VU.length deltas 
                                    then deltas VU.! neuronIdx 
                                    else 0.0
                        in (weights neuron VU.! inputIdx) * delta
                    ) 
                    (neurons layer)
            in V.sum contributions
    in (gradients, nextDeltas)

-- Fixed convLayerBackprop function with proper dimension handling
convLayerBackprop :: LearningRate -> V.Vector ConvFilter -> VU.Vector Delta -> V.Vector (VU.Vector Weight)
convLayerBackprop !lr !filters !deltas =
    let !numFilters = V.length filters
        !filterSize = VU.length $ V.head $ filterWeights $ V.head filters
    in V.generate numFilters $ \i ->
        let !filter = filters V.! i
            !gradients = VU.generate filterSize $ \j ->
                let !delta = if i < VU.length deltas && j < VU.length deltas
                            then deltas VU.! j
                            else 0.0
                in delta * lr
        in gradients

clipValue :: Double -> Double
clipValue x
    | isNaN x = 0.0
    | isInfinite x = signum x * 10.0
    | abs x > 10.0 = signum x * 10.0
    | otherwise = x
{-# INLINE clipValue #-}

-- Modify updateFilter to use clipping
updateFilter :: LearningRate -> ConvFilter -> VU.Vector Weight -> ConvFilter
updateFilter !lr !filter !grads =
    let !currentWeights = V.head $ filterWeights filter
        !newWeights = VU.generate (VU.length currentWeights) $ \i ->
            let !grad = if i < VU.length grads then grads VU.! i else 0.0
                !newWeight = (currentWeights VU.! i) - grad
            in clipValue newWeight
    in filter { filterWeights = V.singleton newWeights }

-- Modify updateLayer to use clipping
updateLayer :: LearningRate -> Layer -> V.Vector (VU.Vector Weight, Bias) -> Layer
updateLayer !lr !layer !grads =
    let !numNeurons = V.length $ neurons layer
        !newNeurons = V.generate numNeurons $ \i ->
            let !neuron = neurons layer V.! i
                (!weightGrads, !biasGrad) = 
                    if i < V.length grads 
                    then grads V.! i
                    else (VU.replicate (VU.length $ weights neuron) 0.0, 0.0)
                !newWeights = VU.map clipValue $ 
                    VU.zipWith (-) (weights neuron) weightGrads
                !newBias = clipValue $ bias neuron - biasGrad
            in Neuron newWeights newBias
    in layer { neurons = newNeurons }

loadMNISTData :: FilePath -> FilePath -> IO MNISTData
loadMNISTData !imagesPath !labelsPath = do
    !imageData <- BS.readFile imagesPath
    !labelData <- BS.readFile labelsPath
    let !images = parseIDXImages imageData
        !labels = parseIDXLabels labelData
        !oneHotLabels = V.map oneHotEncode labels
    return $! MNISTData images oneHotLabels

parseIDXImages :: BS.ByteString -> V.Vector (VU.Vector Double)
parseIDXImages !bs =
    let !header = BS.take 16 bs
        !imageData = BS.drop 16 bs
        !numImages = fromIntegral $ runGet getWord32be $ BSL.fromStrict $ BS.take 4 $ BS.drop 4 header
        !rows = 28
        !cols = 28
    in V.generate numImages $ \i ->
        VU.generate (rows * cols) $ \j ->
            fromIntegral (BS.index imageData (i * rows * cols + j)) / 255.0

parseIDXLabels :: BS.ByteString -> V.Vector Int
parseIDXLabels !bs =
    let !labelData = BS.drop 8 bs
    in V.generate (BS.length labelData) $ \i ->
        fromIntegral $ BS.index labelData i

oneHotEncode :: Int -> VU.Vector Double
oneHotEncode !label = VU.generate 10 $ \i -> if i == label then 1.0 else 0.0

monitorTraining :: CNN -> VU.Vector Double -> VU.Vector Double -> IO ()
monitorTraining cnn input target = do
    let (output, _) = forwardProp cnn input
    putStrLn $ "Target: " ++ show (VU.toList target)
    putStrLn $ "Output: " ++ show (VU.toList output)
    putStrLn $ "Error: " ++ show (VU.sum $ VU.zipWith (-) target output)
    putStrLn $ "Output Layer Neurons: " ++ show (V.length $ neurons $ outputLayer cnn)
    putStrLn $ "FC Layer Neurons: " ++ show (V.length $ neurons $ fullyConnected cnn)
    putStrLn $ "Conv2 Filters: " ++ show (V.length $ convLayer2 cnn)
    putStrLn $ "Conv1 Filters: " ++ show (V.length $ convLayer1 cnn)

trainBatch :: LearningRate -> CNN -> [(VU.Vector Double, VU.Vector Double)] -> IO CNN
trainBatch !lr !cnn !batch = do
    return $! foldl' (\acc (input, target) ->
        let !cnnUpdated = backprop lr acc input target
        in cnnUpdated) cnn batch

train :: LearningRate -> Int -> Int -> CNN -> MNISTData -> IO CNN
train !lr !epochs !batchSize !initialCNN !mnistData = do
    let !trainingPairs = V.zip (images mnistData) (labels mnistData)
        !numBatches = V.length trainingPairs `div` batchSize
        !totalSteps = epochs * numBatches
    
    foldM (\cnn epoch -> do
        printf "\nEpoch %d/%d\n" epoch epochs
        printf "["
        hFlush stdout
        
        -- Add monitoring for first sample of first batch
        when (epoch == 1) $ do
            let (firstInput, firstTarget) = V.head trainingPairs
            monitorTraining cnn firstInput firstTarget
        
        let !batches = [V.toList $ V.slice (i * batchSize) batchSize trainingPairs 
                      | i <- [0..numBatches-1]]
        
        trainedEpochCNN <- foldM (\cnn' (batchNum, batch) -> do
            when (batchNum `mod` (numBatches `div` 50) == 0) $ do
                let progress = ((epoch - 1) * numBatches + batchNum) * 100 `div` totalSteps
                printf "\rEpoch %d/%d [%s%s] %d%%" 
                    epoch epochs 
                    (replicate (progress `div` 2) '=')
                    (replicate (50 - (progress `div` 2)) ' ')
                    progress
                hFlush stdout
            
            !trainedCNN <- trainBatch lr cnn' batch
            return $! trainedCNN) cnn (zip [1..] batches)
            
        return $! trainedEpochCNN) initialCNN [1..epochs]

saveCNN :: FilePath -> CNN -> IO ()
saveCNN path cnn = do
    let encoded = Binary.encode cnn
    BSL.writeFile path encoded
    putStrLn $ "Model saved successfully to " ++ path
    putStrLn $ "Model size: " ++ show (BSL.length encoded) ++ " bytes"

loadCNN :: FilePath -> IO (Either String CNN)
loadCNN path = do
    currentDir <- getCurrentDirectory
    putStrLn $ "Current directory: " ++ currentDir
    putStrLn $ "Looking for model at: " ++ path
    exists <- doesFileExist path
    if not exists
        then return $ Left $ "Model file not found: " ++ path
        else do
            content <- BSL.readFile path
            case Binary.decodeOrFail content of
                Left (_, _, err) -> return $ Left $ "Failed to decode model: " ++ err
                Right (_, _, cnn) -> return $ Right cnn

-- Helper function to ensure numeric validity
validateVec :: String -> VU.Vector Double -> VU.Vector Double
validateVec name vec = 
    let !maxVal = VU.maximum vec
        !minVal = VU.minimum vec
        !hasNaN = VU.any isNaN vec
        !hasInf = VU.any isInfinite vec
        !validVec = VU.map (\x -> if isNaN x || isInfinite x then 0.0 else x) vec
    in unsafePerformIO $ do
        when hasNaN $ putStrLn $ name ++ ": Contains NaN values"
        when hasInf $ putStrLn $ name ++ ": Contains Infinite values"
        putStrLn $ name ++ " range: [" ++ show minVal ++ ", " ++ show maxVal ++ "]"
        return validVec

-- Modified processImage to validate intermediate results
processImage :: VU.Vector Double -> CNN -> (VU.Vector Double, VU.Vector Double)
processImage !input !cnn = unsafePerformIO $ do
    putStrLn $ "Input size: " ++ show (VU.length input)
    putStrLn $ "Input range: [" ++ show (VU.minimum input) ++ ", " ++ show (VU.maximum input) ++ "]"
    
    -- First convolution layer
    let !conv1Outputs = V.map (convolve input) (convLayer1 cnn)
        !conv1Output = validateVec "Conv1" $ VU.concat $ V.toList conv1Outputs
    
    putStrLn $ "Conv1 size: " ++ show (VU.length conv1Output)
    
    -- First max pooling
    let !pooled1 = validateVec "Pool1" $ maxPool conv1Output
    
    putStrLn $ "Pool1 size: " ++ show (VU.length pooled1)
    
    -- Second convolution layer
    let !conv2Outputs = V.map (convolve pooled1) (convLayer2 cnn)
        !conv2Output = validateVec "Conv2" $ VU.concat $ V.toList conv2Outputs
    
    putStrLn $ "Conv2 size: " ++ show (VU.length conv2Output)
    
    -- Second max pooling
    let !pooled2 = validateVec "Pool2" $ maxPool conv2Output
    
    putStrLn $ "Pool2 size: " ++ show (VU.length pooled2)
    
    -- Final predictions
    let (!predictions, _) = forwardProp cnn input
        !validPredictions = validateVec "Predictions" predictions
    
    -- Force evaluation and return
    return $! (conv1Output, validPredictions)

-- Modified interactiveMode to ensure proper JSON handling
interactiveMode :: CNN -> IO ()
interactiveMode !cnn = do
    hSetBuffering stdin LineBuffering
    hSetBuffering stdout LineBuffering
    
    putStrLn "Interactive mode started... waiting for input"
    hFlush stdout
    
    forever $ do
        putStrLn "Waiting for input..."
        hFlush stdout
        
        input <- getLine
        putStrLn $ "Received input (first 100 chars): " ++ take 100 input
        hFlush stdout
        
        case Aeson.decode (BSL.pack input) of
            Nothing -> do
                putStrLn $ "Error: Invalid JSON input format: " ++ take 100 input
                hFlush stdout
            
            Just imgData -> do
                putStrLn "Successfully decoded input array"
                putStrLn $ "Input array length: " ++ show (VU.length imgData)
                hFlush stdout
                
                let (!layerOutput, !predictions) = processImage imgData cnn
                putStrLn $ "Layer output size: " ++ show (VU.length layerOutput)
                putStrLn $ "Predictions size: " ++ show (VU.length predictions)
                putStrLn $ "Sample predictions: " ++ show (take 5 $ VU.toList predictions)
                hFlush stdout
                
                let result = object
                        [ fromString "layer_output" .= VU.toList layerOutput
                        , fromString "predictions" .= VU.toList predictions
                        ]
                
                putStrLn "Encoding response..."
                let encoded = Aeson.encode result
                putStrLn $ "Response size: " ++ show (BSL.length encoded) ++ " bytes"
                
                BSL.putStr encoded
                putStr "\n"
                hFlush stdout
                
                putStrLn "Response sent"
                hFlush stdout

main = do
    args <- getArgs
    case args of
        ["train"] -> do
            putStrLn "Loading MNIST data..."
            !mnistData <- loadMNISTData "data/train-images.idx3-ubyte" 
                                      "data/train-labels.idx1-ubyte"
            
            -- Add debug information
            putStrLn $ "First image dimensions: " ++ show (VU.length $ V.head $ images mnistData)
            putStrLn $ "Number of training samples: " ++ show (V.length $ images mnistData)
            
            putStrLn "Initializing CNN..."
            !cnn <- initCNN
            
            -- Test first forward pass
            let firstImage = V.head $ images mnistData
                (output, _) = forwardProp cnn firstImage
            putStrLn $ "First forward pass output dimensions: " ++ show (VU.length output)
            
            putStrLn "Starting training..."
            !trainedCNN <- train 0.001 8 64 cnn mnistData
            
            putStrLn "\nSaving model..."
            saveCNN "trained_model.cnn" trainedCNN
            putStrLn "Training complete!"
        
        [] -> do
            modelResult <- loadCNN "trained_model.cnn"
            case modelResult of
                Left err -> putStrLn $ "Error: " ++ err
                Right cnn -> interactiveMode cnn