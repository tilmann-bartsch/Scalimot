name := "Scalimot"

scalaVersion := "2.13.6"
scalacOptions += "-language:higherKinds"
// addCompilerPlugin("org.typelevel" %% "kind-projector" % "0.13.0" cross CrossVersion.full)

//scalacOptions += "-Ydelambdafy:inline"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "1.2",
  "org.scalanlp" %% "breeze-natives" % "1.2",
)
