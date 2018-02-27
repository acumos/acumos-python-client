
# Release Notes

## v0.5
* Modeling
	* Python 3.6 NamedTuple syntax support now tested
	* User documentation includes example of new NamedTuple syntax
* Model wrapper
	* Model wrapper now has APIs for consuming and producing Python dicts and JSON strings
* Protobuf and protoc
	* An explicit check for protoc is now made, which raises a more informative error message
	* User documentation is more clear about dependence on protoc, and provides an easier way to install protoc via Anaconda
* Keras
	* The active keras backend is now included as a tracked module
	* keras_contrib layers are now supported

## v0.4

* Replaced library-specific onboarding functions with "new-style" models
    * Support for arbitrary Python functions using type hints
    * Support for custom user-defined types
    * Support for TensorFlow models
    * Improved dependency introspection
    * Improved object serialization mechanisms
