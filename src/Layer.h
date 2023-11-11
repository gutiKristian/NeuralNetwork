
template <typename Derived>
class Layer
{
public:
	Layer() = default;

	void forward()
	{
		Derived& derived = static_cast<Derived&>(*this);

	}
};

class InputLayer : public Base<InputLayer>
{
public:
	InputLayer() = default;

};