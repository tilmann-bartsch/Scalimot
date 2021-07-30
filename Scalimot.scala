/* Scalimot. Scala implemenation of LoLiMoT (Local Linear Model Tree)
*/

import breeze.linalg.{DenseVector, DenseMatrix, Axis, sum, max, min}
import breeze.numerics.{pow, sqrt, exp}
import breeze.stats.distributions.Uniform

object Scalimot extends App {

  /** Class describing a model (which predicts a label from given features).
  *  
  *  @param q Double containing the sum of the squared distanced of model
  *         predictions and label on training data.
  *
  *  @param predict function that takes a vector of features and predict the
  *         label.
  */ 
  case class Model(q: Double, predict: DenseVector[Double] => Double)
  def emptyModel(): Model = Model(Double.PositiveInfinity, _ => Double.NaN)
  
 /** Perform linear regression on data and return a fitted Model.
  *
  *  @param data Breeze DenseMatrix[Double] containing data points. First
  *         column contains labels. Other columsn contain features.
  */
  def linearRegression(data: DenseMatrix[Double]): Model = {
    val labels = data(::, 0)          // DenseVector containing labels
    val features = data(::, 1 to -1)  // DenseMatrix contatining features
    
   /** Perform Gradient Descent for a linear model.
    *
    *  @param a breeze DenseVector containing the information of the linear model predicting.
    *         
    *         For label y and features x_1, ..., x_k and a = (a_0, a_1, ..., a_k) the linear model is
    *           y = a_0 + a_1*x_1 + ... + a_k*x_k
    *  @param n number of remaining steps
    *  @return breeze DenseVector encoding updated linear model
    */
    @annotation.tailrec
    def gradientStep(a: DenseVector[Double], n: Int): DenseVector[Double] = 
    {
      val N = data.rows                 // Number of Data Points
            
      // Predict labels with features with the linear model given by a
      val pred = a(0) -  (-features*a(1 to -1))
      
      // Calculate gradient of 1/N*sum(pow(label - pred, 2)) with respect to a
      val g0 = DenseVector(sum(-2.0/N * (labels - pred)))
      val gs = -2.0/N * features.t * (labels - pred)
      val grad = DenseVector.vertcat(g0, gs)
      
      // Update a
      val L = 0.01  // Learning rate
      val new_a = a - L * grad
      
      // Stop computation, if gradient is very small or no remaining steps left
      // otherwise repeat compuation with new_a
      if (sum(sqrt(pow(grad, 2))) < 1e-12) new_a  
      else if (n<=1) new_a
      else gradientStep(new_a, n-1)
    }
    
    val a_start = DenseVector.zeros[Double](data.cols) // Start with the linear function 
                                                       // which maps everything to zero.
    val a = gradientStep(a_start, 2500)                // Gradient Descent 
                                                       // (gradientStep is recursive)
    
    val pred = a(0) -  (-features*a(1 to -1)) // Predict label with features and calculated a
    val q = sum(pow(labels - pred, 2))        // q = sum of squared distances of labels and
                                              // predictions.
                         
    Model(q, (inp) => a(0) - (-a(1 to - 1) dot inp)) // Return Model
  }
  
 /** Make a model using the Local Model Tree concept.
  *
  * @param data Breeze DenseMatrix[Double] object containg the data
  *        (first column: label, other columns: features
  *
  * @param baseModel Function turning input data to a model. The model
  *        construction of this is used as the models at the leaf of the tree.
  * @param splits:  Number of of 'model split' to perform
  *
  * @return Instance of class Model obtained by LoLiMoT algorithm
  */
  def makeModel(data: DenseMatrix[Double], 
            baseModel: DenseMatrix[Double] => Model,
            splits: Int):
  Model =
  {
   /* Define simple binary tree structure containing value of
    * type A at every leaf and value of Type B at every branch. */
    sealed trait Tree[+A, +B] {
     /* Fold only considers values at branches */
      def fold[C](f: A => C)(g: (C,C) => C): C = this match {
        case Leaf(a) => f(a)
        case Branch(_,l,r) => g(l.fold(f)(g), r.fold(f)(g))
      }
    }      
    case class Leaf[A, B](value: A) extends Tree[A, B]
    case class Branch[A, B](value: B, left: Tree[A, B], right: Tree[A, B]) extends Tree[A, B]
    
   /* Define Tree Strucuture `ModelTree` which represents the local linear model tree.
    *
    * A ModelTree stores a model at its leaf and a function f = DenseVector[Double] => Dir
    * at every branch. For a branch(f, t1, t2) f decides if features given by a vector should
    * be predicted by t1 or t2. findAndPredict applies these decision functions recursively
    * to obtain the prediction of a ModelTree for features given by a vector
    *
    * Maximum and sum provide the maximum and sum of the q values of models at all leafs
    * respectively.
    */
    type BranchData = DenseVector[Double] => (Double, Double)    
    type LeafData = Model
    type ModelTree = Tree[LeafData, BranchData]
    def findAndPredict(tree: ModelTree, vec: DenseVector[Double]): Double =
      tree match {
        case Leaf(model) => model.predict(vec)
        case Branch(f, t1, t2) => {
          val (d1, d2) = f(vec)
          d1*findAndPredict(t1, vec) + d2*findAndPredict(t2, vec)
        }
      }
    def maximum(t: ModelTree): Double = 
            t.fold[Double](a => a.q)(_ max _)
    def sum(t: ModelTree): Double = 
            t.fold[Double](a => a.q)(_ + _)
      
   /** Split Leaf of a ModelTree multiple times by 
    *   1. Searching for the leaf containing the model which has
    *      the highest q value.
    *   2. Replacing that leaf by a Branch(f, Leaf(mod1), Leaf(mod2))
    *      structure. This new structure is constructed by the function
    *      findBestSplit.
    * To construct a ModelTree starting from a model m call i.e.
    *   splitLeaf(Leaf(m), 5)
    *      
    * @param ModelTree Tree representing the current model which is to be split
    *        spl times.
    * @param spl number of splits to perform
    */
    @annotation.tailrec
    def splitLeafs(tree: ModelTree, spl: Int): 
            ModelTree = {
      // Make no more splits if spl equals 0
      if (spl<=0) return tree
      
     /*Recursive Function traveling the tree while keeping the data which belongs
       to the current branch or leaf (as given by the function stored at branch).*/
      def go(tree : Tree[LeafData, BranchData],
             local_data: DenseMatrix[Double]):
          ModelTree =
      tree match {
        case Branch(f, left, right) => {
          val max_l = maximum(left)
          val max_r = maximum(right)
          if (max_l > max_r) {
            var vectors = 
                for (i <- 0 until local_data.rows
                     if f(local_data(i, 1 to -1).t)._1 > 0.5)
                        yield local_data(i,::).t.toDenseMatrix
            val filt_data = DenseMatrix.vertcat(vectors:_*)            
            Branch(f, go(left, filt_data), right) }
          else {
            var vectors = 
                for (i <- 0 until local_data.rows 
                     if f(local_data(i, 1 to -1).t)._2 > 0.5)
                        yield local_data(i,::).t.toDenseMatrix
            val filt_data = DenseMatrix.vertcat(vectors:_*)
            Branch(f, left, go(right, filt_data)) }
        }
        case Leaf(ld) => {
         /* Test different splits of Data at the current Leaf by
          * by dividing the local data according to different features.
          *
          * Compare the current split to previous split and keep
          * the one with better sum of q values.
          */
          @annotation.tailrec
          def findBestSplit(t: ModelTree, feature_dim: Int):
          ModelTree = {
            // Stop if all feature dims have been tested
            if (feature_dim < 0) return t
            
            // Obtain all values of "feature_dim"th feature and
            // calculate their mean value as threshold.
            val vec = local_data(::, feature_dim+1)
            val threshold = (max(vec) + min(vec)) / 2
            
            // Obtain indices of data points (rows in data) with
            // "feature_dim"th feature lying left and right of
            // the threshold.
            val inds_left = vec.findAll(_ <  threshold)
            val inds_right = vec.findAll(_ >= threshold)
            
            // Construct new ModelTree new_t = Branch(f, mod1, mod2) as a 
            // replacment candidate of the current leaf ld.
            val b = threshold; val s = 0.001
            val f = (v: DenseVector[Double]) => {
                val x = v(feature_dim)
                ( 1 / (1 + exp((x-b)/s)) , 1 / (1 + exp(-(x-b)/s)) )
            }
            
            val new_mod_l = baseModel(local_data(inds_left, ::).toDenseMatrix)
            val new_mod_r = baseModel(local_data(inds_right, ::).toDenseMatrix)
            val new_t = Branch(f, Leaf(new_mod_l), Leaf(new_mod_r))
            
            // If the sum of q values of the new split is smaller than the
            // old one then replace the split. Otherwise keep the old one.
            if (sum(new_t) < sum(t)) {
              findBestSplit(new_t, feature_dim-1)
            }
            else
              findBestSplit(t, feature_dim-1)            
          }
          
          // Create init = Branch(f,m1,m2) representing empty Model with sum of q values Infinity
          val mod1 = emptyModel()
          val mod2 = emptyModel()
          val init = Branch((_: DenseVector[Double]) => (0.0,1.0), Leaf(mod1), Leaf(mod2))
          // local_data.cols = 1 + number of features, such that in recursive call of 
          // findBestSplit: feature_dim = ...,2,1,0
          findBestSplit(init, local_data.cols - 2)
        }
      }
      splitLeafs(go(tree, data), spl-1)
    }
    
    // Make ModelTree init containing only a Leaf with the Model obtained
    // by baseModel
    val init = Leaf(baseModel(data))
    // Build model tree
    val modelTree = splitLeafs(init, splits)
    // Return model using the modeltree
    Model(sum(modelTree), (vec) => findAndPredict(modelTree, vec))      
  }

  /**** Simple Test ****/
  
  // Test data
  val features: Int = 2
  val data_points: Int = 10000
  // Create matrix. Every row is data point. First column contains
  // labels. Other columns contain features.
  var data = DenseMatrix.zeros[Double](data_points, features+1)
  // Set features to random numbers between -1 and 1
  val dist = Uniform(-1,1)
  data(::, 1 to -1) := DenseMatrix.rand(data_points, features, dist)
  // Set label as the quadratic sum of the features
  data(::, 0) := sum(pow(data(::, 1 to -1), 2), Axis._1) + 0.0001 * DenseVector.rand(data_points, dist)
  
  // Test different amount of model splits
  for (splits <- 0 to 2){
    val mod = makeModel(data, linearRegression, splits)
    println("Splits: %d, q = %e".format(splits, mod.q/data.rows))
    println()
  }
  
  // Test model prediction
  val splits = 4
  val mod = makeModel(data, linearRegression, splits)  
  print("Inference for Model with:  ")
  println("Splits: %d, q = %e".format(splits, mod.q/data.rows))
  println("Predicions, True Value, Features")
  for (_ <- 0 to 5){
    val x = DenseVector.rand(features, dist)
    val y = mod.predict(x)
    println("%10.5f, %10.5f, %s".format(y, sum(pow(x,2)), x))
  }
}
