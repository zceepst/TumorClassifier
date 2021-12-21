module Viz

using LaTeXStrings
using Plots

# values extracted using awk, update to pipe results from submodules
function binaryLeNet5Results()
	train_loss = [0.6969f0,0.4488f0,0.3779f0,0.3007f0,0.2504f0,0.2306f0,0.2014f0,0.1927f0,0.1888f0,0.1667f0,0.1526f0]
	train_acc = [0.4208,0.8512,0.8512,0.864,0.8904,0.9024,0.9152,0.9216,0.9188,0.9376,0.9416]
	test_loss = [0.6957f0,0.475f0,0.4039f0,0.3211f0,0.2644f0,0.2491f0,0.2259f0,0.2229f0,0.2192f0,0.2037f0,0.197f0]
	test_acc = [0.442,0.836,0.836,0.846,0.886,0.898,0.928,0.928,0.918,0.936,0.936]
	epochs = 0:10
	fig = plot(
		epochs, train_loss;
		title=L"Binary\ LeNet5\ validated\ performance",
		xlabel=L"Epochs",
		ylabel=L"Loss/Accuracy",
		label=L"Train\ loss",
		leg=:right,
		linecolor=:red,
		lw=2
	)
	plot!(
		fig, epochs, train_acc;
		label=L"Train\ accuracy",
		linecolor=:orange
	)
	plot!(
		fig, epochs, test_acc;
		label=L"Test\ accuracy",
		linecolor=:blue
	)
	plot!(
		fig, epochs, test_loss;
		label=L"Test\ loss",
		linecolor=:green
	)
	savefig(fig, "output/binaryCNNPerf.png")
end

function multiLeNet5Results()
	train_loss = [1.3821f0,1.1531f0,0.9997f0,0.9058f0,0.8222f0,0.7607f0,0.7295f0,0.6844f0,0.6452f0,0.7089f0,0.5978f0,0.5626f0,0.5451f0,0.549f0,0.5252f0,0.496f0,0.4751f0,0.4959f0,0.4127f0,0.3958f0,0.3783f0]
	train_acc = [0.2956,0.5036,0.6184,0.6412,0.6568,0.7176,0.7192,0.7364,0.7624,0.726,0.7832,0.796,0.7948,0.794,0.794,0.8276,0.8148,0.8104,0.8608,0.8604,0.8624]
	test_loss = [1.3839f0,1.1931f0,1.0527f0,0.9751f0,0.9008f0,0.8273f0,0.8124f0,0.7758f0,0.7346f0,0.7887f0,0.6999f0,0.6693f0,0.671f0,0.6559f0,0.6737f0,0.6253f0,0.628f0,0.6264f0,0.5785f0,0.5631f0,0.5662f0]
	test_acc = [0.288,0.482,0.574,0.602,0.630,0.698,0.652,0.688,0.73,0.678,0.724,0.754,0.734,0.764,0.74,0.756,0.754,0.772,0.772,0.778,0.78]
	@assert length(train_loss) == length(test_loss)
	@assert length(train_acc) == length(test_acc)
	epochs = 0:20
	fig= plot(
		epochs, train_loss;
		title=L"Multiclass\ LeNet5\ validated\ performance",
		xlabel=L"Epochs",
		ylabel=L"Loss/Accuracy",
		label=L"Train\ loss",
		leg=:topright,
		linecolor=:red,
		lw=2
	)
	plot!(
		fig, epochs, train_acc;
		label=L"Train\ accuracy",
		linecolor=:orange
	)
	plot!(
		fig, epochs, test_loss;
		label=L"Test\ loss",
		linecolor=:green
	)
	plot!(
		fig, epochs, test_acc;
		label=L"Test\ accuracy",
		linecolor=:blue
	)
	savefig(fig, "output/multiCNNPerf.png")
end

end
