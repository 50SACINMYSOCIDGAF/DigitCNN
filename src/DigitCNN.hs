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
import Data.Aeson (ToJSON(..), FromJSON(..), object, (.=), Key)
import GHC.Generics
import Control.DeepSeq
import System.Directory (doesFileExist)
import Text.Printf
import Control.Concurrent
import Control.Concurrent.Async
import System.Environment (getArgs)
import Data.Aeson (ToJSON, FromJSON, encode, decode, object, (.=))
import Control.Monad (when)
import Control.Concurrent (threadDelay)
import qualified Data.Text as T
import Data.String (fromString)

-- Type aliases for clarity and performance
type Weight = Double
type Bias = Double
type Activation = Double
type Delta = Double
type LearningRate = Double

-- Neural network structures
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

data CNN = CNN {
    convLayer1 :: !(V.Vector ConvFilter),
    convLayer2 :: !(V.Vector ConvFilter),
    fullyConnected :: !Layer,
    outputLayer :: !Layer
} deriving (Show, Generic)

instance NFData CNN

instance Binary.Binary Neuron where
    put (Neuron w b) = Binary.put (VU.toList w) >> Binary.put b
    get = do
        w <- VU.fromList <$> Binary.get
        b <- Binary.get
        return $ Neuron w b

instance Binary.Binary Layer where
    put (Layer n a) = Binary.put (V.toList n) >> Binary.put (VU.toList a)
    get = do
        n <- V.fromList <$> Binary.get
        a <- VU.fromList <$> Binary.get
        return $ Layer n a

instance Binary.Binary ConvFilter where
    put (ConvFilter w b) = Binary.put (map VU.toList $ V.toList w) >> Binary.put b
    get = do
        w <- V.fromList . map VU.fromList <$> Binary.get
        b <- Binary.get
        return $ ConvFilter w b

instance Binary.Binary CNN where
    put (CNN c1 c2 fc out) = Binary.put (V.toList c1) >> Binary.put (V.toList c2) >> Binary.put fc >> Binary.put out
    get = CNN <$> (V.fromList <$> Binary.get) <*> (V.fromList <$> Binary.get) <*> Binary.get <*> Binary.get

instance ToJSON (VU.Vector Double) where
    toJSON = Aeson.toJSON . VU.toList

instance FromJSON (VU.Vector Double) where
    parseJSON = fmap VU.fromList . Aeson.parseJSON

-- Training data structure
data MNISTData = MNISTData {
    images :: !(V.Vector (V.Vector (VU.Vector Double))),
    labels :: !(V.Vector (VU.Vector Double))
}

-- Activation functions
sigmoid :: Double -> Double
sigmoid !x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}

sigmoidDerivative :: Double -> Double
sigmoidDerivative !x = s * (1 - s)
  where
    !s = sigmoid x
{-# INLINE sigmoidDerivative #-}

-- Optimized convolution operation
convolve :: V.Vector (VU.Vector Double) -> ConvFilter -> V.Vector (VU.Vector Double)
convolve !input !filter = V.generate outputHeight $ \i ->
    VU.generate outputWidth $ \j ->
        let !result = sum [sum [((input V.! (i + k)) VU.! (j + l)) * 
                               ((filterWeights filter V.! k) VU.! l)
                          | l <- [0..filterSize-1]
                          ] 
                    | k <- [0..filterSize-1]
                    ] + filterBias filter
        in result
  where
    !inputHeight = V.length input
    !inputWidth = VU.length (input V.! 0)
    !filterSize = V.length (filterWeights filter)
    !outputHeight = inputHeight - filterSize + 1
    !outputWidth = inputWidth - filterSize + 1
{-# INLINE convolve #-}

-- Max pooling with unboxed vectors
maxPool :: V.Vector (V.Vector (VU.Vector Double)) -> V.Vector (V.Vector (VU.Vector Double))
maxPool !input = V.generate outputHeight $ \i ->
    V.generate outputWidth $ \j ->
        let values = [((input V.! (i * 2 + di)) V.! (j * 2 + dj))
                    | di <- [0,1], dj <- [0,1]]
        in maximum values
  where
    !inputHeight = V.length input
    !inputWidth = V.length (input V.! 0)
    !outputHeight = inputHeight `div` 2
    !outputWidth = inputWidth `div` 2
{-# INLINE maxPool #-}

-- Flatten operation optimized for unboxed vectors
flatten :: V.Vector (VU.Vector Double) -> VU.Vector Double
flatten !matrix = VU.concat $ V.toList matrix
{-# INLINE flatten #-}

-- Forward propagation through a layer
layerForward :: Layer -> VU.Vector Double -> Layer
layerForward !layer !input = layer { activations = newActivations }
  where
    !newActivations = VU.generate (V.length $ neurons layer) $ \i ->
        let !neuron = neurons layer V.! i
            !sum = VU.sum $ VU.zipWith (*) (weights neuron) input
        in sigmoid (sum + bias neuron)
{-# INLINE layerForward #-}

-- Initialize random values
randomWeight :: IO Double
randomWeight = randomRIO (-0.5, 0.5)
{-# INLINE randomWeight #-}

-- Initialize CNN components
randomConvFilter :: Int -> IO ConvFilter
randomConvFilter !size = do
    !weights <- V.replicateM size $ VU.replicateM size randomWeight
    !bias <- randomWeight
    return $! ConvFilter weights bias

randomLayer :: Int -> Int -> IO Layer
randomLayer !numNeurons !inputSize = do
    !neurons <- V.replicateM numNeurons $ do
        !ws <- VU.replicateM inputSize randomWeight
        !b <- randomWeight
        return $! Neuron ws b
    return $! Layer neurons (VU.replicate numNeurons 0.0)

initCNN :: IO CNN
initCNN = do
    !conv1 <- V.replicateM 6 (randomConvFilter 5)
    !conv2 <- V.replicateM 16 (randomConvFilter 5)
    !fc <- randomLayer 120 400
    !output <- randomLayer 10 120
    return $! CNN conv1 conv2 fc output

-- Forward propagation through the entire network
forwardProp :: CNN -> V.Vector (VU.Vector Double) -> (VU.Vector Double, CNN)
forwardProp !cnn !input = 
    let !conv1Output = V.map (convolve input) (convLayer1 cnn)
        !pooled1 = maxPool conv1Output
        !conv2Output = V.map (convolve pooled1) (convLayer2 cnn)
        !pooled2 = maxPool conv2Output
        !flattened = flatten pooled2
        !fcOutput = layerForward (fullyConnected cnn) flattened
        !outputResult = layerForward (outputLayer cnn) (activations fcOutput)
    in (activations outputResult, 
        cnn { fullyConnected = fcOutput, outputLayer = outputResult })

-- Backpropagation
backprop :: LearningRate -> CNN -> V.Vector (VU.Vector Double) -> VU.Vector Double -> CNN
backprop !lr !cnn !input !target =
    let (!output, !forwardState) = forwardProp cnn input
        !outputError = VU.zipWith (-) output target
        
        -- Output layer gradients
        (!outputGrads, !fcError) = layerBackprop lr (outputLayer forwardState) outputError
        (!fcGrads, !conv2Error) = layerBackprop lr (fullyConnected forwardState) fcError
        
        -- Convolution layer gradients
        !conv2Grads = convLayerBackprop lr (convLayer2 forwardState) conv2Error
        !conv1Grads = convLayerBackprop lr (convLayer1 forwardState) conv2Grads
        
        -- Apply all gradients
        !newCnn = cnn {
            convLayer1 = applyConvGrads lr (convLayer1 cnn) conv1Grads,
            convLayer2 = applyConvGrads lr (convLayer2 cnn) conv2Grads,
            fullyConnected = applyGrads lr (fullyConnected cnn) fcGrads,
            outputLayer = applyGrads lr (outputLayer cnn) outputGrads
        }
    in newCnn

-- Layer backpropagation
layerBackprop :: LearningRate -> Layer -> VU.Vector Delta -> 
                 (V.Vector (VU.Vector Weight, Bias), VU.Vector Delta)
layerBackprop !lr !layer !deltas = (gradients, nextDeltas)
  where
    !gradients = V.zipWith computeGradients (neurons layer) (VU.toList deltas)
    !nextDeltas = VU.generate (VU.length $ weights $ V.head $ neurons layer) $ \i ->
        VU.sum $ VU.zipWith (*) (VU.fromList $ V.toList $ 
            V.map (\n -> (weights n) VU.! i) (neurons layer)) deltas
    
    computeGradients neuron delta =
        let !weightGrads = VU.map (* delta) (weights neuron)
            !biasGrad = delta
        in (weightGrads, biasGrad)

-- Convolution layer backpropagation
convLayerBackprop :: LearningRate -> V.Vector ConvFilter -> V.Vector Delta ->
                     (V.Vector (V.Vector (VU.Vector Weight), Bias), V.Vector Delta)
convLayerBackprop !lr !filters !deltas = (gradients, nextDeltas)
  where
    gradients = V.map computeFilterGrads filters
    nextDeltas = V.replicate (V.length filters) (VU.replicate (VU.length deltas) 0)
    
    computeFilterGrads filter =
        let !weightGrads = V.map (VU.map (* lr)) (filterWeights filter)
            !biasGrad = lr * VU.sum deltas
        in (weightGrads, biasGrad)

-- Apply gradients to layers
applyGrads :: LearningRate -> Layer -> V.Vector (VU.Vector Weight, Bias) -> Layer
applyGrads !lr !layer !grads =
    layer { neurons = V.zipWith updateNeuron (neurons layer) grads }
  where
    updateNeuron neuron (weightGrads, biasGrad) =
        Neuron {
            weights = VU.zipWith (\w g -> w - lr * g) (weights neuron) weightGrads,
            bias = bias neuron - lr * biasGrad
        }

applyConvGrads :: LearningRate -> V.Vector ConvFilter -> V.Vector (V.Vector (VU.Vector Weight), Bias) -> 
                  V.Vector ConvFilter
applyConvGrads !lr !filters !grads =
    V.zipWith updateFilter filters grads
  where
    updateFilter filter (weightGrads, biasGrad) =
        ConvFilter {
            filterWeights = V.zipWith (VU.zipWith (\w g -> w - lr * g))
                           (filterWeights filter) weightGrads,
            filterBias = filterBias filter - lr * biasGrad
        }

-- MNIST data loading
loadMNISTData :: FilePath -> FilePath -> IO MNISTData
loadMNISTData !imagesPath !labelsPath = do
    !imageData <- BS.readFile imagesPath
    !labelData <- BS.readFile labelsPath
    
    let !images = parseIDXImages imageData
        !labels = parseIDXLabels labelData
        !oneHotLabels = V.map oneHotEncode labels
    
    return $! MNISTData images oneHotLabels

parseIDXImages :: BS.ByteString -> V.Vector (V.Vector (VU.Vector Double))
parseIDXImages !bs =
    let !header = BS.take 16 bs
        !imageData = BS.drop 16 bs
        !numImages = fromIntegral $ runGet getWord32be $ BSL.fromStrict $ BS.take 4 $ BS.drop 4 header
        !rows = fromIntegral $ runGet getWord32be $ BSL.fromStrict $ BS.take 4 $ BS.drop 8 header
        !cols = fromIntegral $ runGet getWord32be $ BSL.fromStrict $ BS.take 4 $ BS.drop 12 header
    in V.generate numImages $ \i ->
        V.generate rows $ \r ->
            VU.generate cols $ \c ->
                fromIntegral (BS.index imageData (i * rows * cols + r * cols + c)) / 255.0

parseIDXLabels :: BS.ByteString -> V.Vector Int
parseIDXLabels !bs =
    let !labelData = BS.drop 8 bs
    in V.generate (BS.length labelData) $ \i ->
        fromIntegral $ BS.index labelData i

oneHotEncode :: Int -> VU.Vector Double
oneHotEncode !label = VU.generate 10 $ \i -> if i == label then 1.0 else 0.0

-- Training functions
trainBatch :: LearningRate -> CNN -> [(V.Vector (VU.Vector Double), VU.Vector Double)] -> IO CNN
trainBatch !lr !cnn !batch = do
    let trainOne !acc !(input, target) =
            let (!_, !cnnForward) = forwardProp acc input
                !cnnUpdated = backprop lr cnnForward input target
            in cnnUpdated
    return $! foldl' trainOne cnn batch

train :: LearningRate -> Int -> Int -> CNN -> MNISTData -> IO CNN
train !lr !epochs !batchSize !initialCNN !mnistData = do
    let !trainingPairs = V.zip (images mnistData) (labels mnistData)
        !numBatches = V.length trainingPairs `div` batchSize
        !totalSteps = epochs * numBatches
        
    foldM (\cnn epoch -> do
        printf "\nEpoch %d/%d\n" epoch epochs
        printf "["
        hFlush stdout
        
        let !batches = [V.slice (i * batchSize) batchSize trainingPairs 
                      | i <- [0..numBatches-1]]
        
        foldM (\cnn' (batchNum, batch) -> do
            when (batchNum `mod` (numBatches `div` 50) == 0) $ do
                let progress = ((epoch - 1) * numBatches + batchNum) * 100 `div` totalSteps
                printf "\rEpoch %d/%d [%s%s] %d%%" 
                    epoch epochs 
                    (replicate (progress `div` 2) '=')
                    (replicate (50 - (progress `div` 2)) ' ')
                    progress
                hFlush stdout
            
            !trainedCNN <- trainBatch lr cnn' (V.toList batch)
            return $! trainedCNN
          ) cnn (zip [1..] batches)  
      ) initialCNN [1..epochs]       

-- Main function with training and interactive modes
main :: IO ()
main = do
    args <- getArgs
    case args of
        ["train"] -> do
            putStrLn "Loading MNIST data..."
            !mnistData <- loadMNISTData "data/train-images-idx3-ubyte" 
                                      "data/train-labels-idx1-ubyte"
            
            putStrLn "Initializing CNN..."
            !cnn <- initCNN
            
            putStrLn "Starting training..."
            !trainedCNN <- train 0.01 10 32 cnn mnistData
            
            putStrLn "Saving model..."
            saveCNN "trained_model.cnn" trainedCNN
            
            putStrLn "Training complete!"
            
        [] -> do
            modelExists <- doesFileExist "trained_model.cnn"
            if not modelExists 
                then putStrLn "Error: Model file not found!"
                else do
                    !cnn <- loadCNN "trained_model.cnn"
                    interactiveMode cnn

-- Save and load functions
saveCNN :: FilePath -> CNN -> IO ()
saveCNN path cnn = BSL.writeFile path (Binary.encode cnn)

loadCNN :: FilePath -> IO CNN
loadCNN path = do
    content <- BSL.readFile path
    case Binary.decode content of
        Nothing -> error "Failed to load model"
        Just cnn -> return cnn

-- Process a single image and return visualization data
processImage :: V.Vector (VU.Vector Double) -> CNN -> 
                ([V.Vector (VU.Vector Double)], VU.Vector Double)
processImage !input !cnn = 
    let !conv1Output = V.map (convolve input) (convLayer1 cnn)
        !pooled1 = maxPool conv1Output
        !conv2Output = V.map (convolve pooled1) (convLayer2 cnn)
        !pooled2 = maxPool conv2Output
        !flattened = flatten pooled2
        !fcOutput = layerForward (fullyConnected cnn) flattened
        (!final, _) = forwardProp cnn input
    in ([conv1Output V.! 0, pooled1 V.! 0, conv2Output V.! 0, pooled2 V.! 0], final)

-- Convert 2D vector to list of lists for JSON
vectorToList :: V.Vector (VU.Vector Double) -> [[Double]]
vectorToList = map VU.toList . V.toList

-- Interactive mode implementation
interactiveMode :: CNN -> IO ()
interactiveMode !cnn = do
    hSetBuffering stdin LineBuffering
    hSetBuffering stdout LineBuffering
    
    forever $ do
        input <- getLine
        case Aeson.decode (BSL.pack input) of
            Nothing -> do
                putStrLn "Error: Invalid input format"
                hFlush stdout
                
            Just imgData -> do
                let (!layerOutputs, !predictions) = processImage imgData cnn
                    result = object [
                        fromString "layer_outputs" .= map vectorToList layerOutputs,
                        fromString "predictions" .= VU.toList predictions
                        ]
                
                BSL.putStr (Aeson.encode result)
                putStr "\n"
                hFlush stdout

-- Utility function for model evaluation
evaluateModel :: CNN -> MNISTData -> IO Double
evaluateModel !cnn !testData = do
    let total = V.length (images testData)
        predictions = V.map (\img -> fst $ forwardProp cnn img) (images testData)
        correct = V.sum $ V.zipWith (\pred target -> 
            if VU.maxIndex pred == VU.maxIndex target then 1 else 0
            ) predictions (labels testData)
    return $ fromIntegral correct / fromIntegral total * 100

-- Error calculation function
calculateError :: VU.Vector Double -> VU.Vector Double -> Double
calculateError !predicted !target =
    VU.sum $ VU.zipWith (\p t -> (p - t) * (p - t)) predicted target

-- Add this to the training function to track progress
printProgress :: Int -> Int -> Int -> Int -> Double -> IO ()
printProgress !epoch !epochs !batch !totalBatches !error = do
    printf "\rEpoch %d/%d - Batch %d/%d - Error: %.4f" 
           epoch epochs batch totalBatches error
    hFlush stdout


