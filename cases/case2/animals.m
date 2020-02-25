%% Example from Neural Networks and Learning Machines of
%  Simon Haykin 3ed, pg.448.

Dove =   [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 1 0];
Hen =    [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0];
Duck =   [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 1];
Goose =  [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 1 1];
Owl =    [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 1 0 1 0];
Hawk =   [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 1 0 1 0];
Eagle =  [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 1 1 0 1 0];
Fox =    [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 1 0 0 0];
Dog =    [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 1 0 0];
Wolf =   [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 1 1 0 1 0 1 1 0 0];
Cat =    [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 1 0 0 0 1 0 0 0];
Tiger =  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 1 0 0 0 1 1 0 0];
Lion =   [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 1 0 1 0 1 1 0 0];
Horse =  [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1 1 1 1 0 0 1 0 0];
Zebra =  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 1 1 1 1 0 0 1 0 0];
Cow =    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 1 1 0 0 0 0 0 0];

source = [Dove;Hen;Duck;Goose;Owl;Hawk;Eagle;Fox; ...
          Dog;Wolf;Cat;Tiger;Lion;Horse;Zebra;Cow];

[w,x,y] = kohonen(100,29,16,source,0.0001);

DoveWinning =  win(w,Dove);
HenWinning = win(w,Hen);
DuckWinning = win(w,Duck);
GooseWinning = win(w,Goose);
OwlWinning = win(w,Owl);
HawkWinning = win(w,Hawk);
EagleWinning = win(w,Eagle);
FoxWinning = win(w,Fox);
DogWinning = win(w,Dog);
WolfWinning = win(w,Wolf);
CatWinning = win(w,Cat);
TigerWinning = win(w,Tiger);
LionWinning = win(w,Lion);
HorseWinning = win(w,Horse);
ZebraWinning = win(w,Zebra);
CowWinning = win(w,Cow);

neighborhood(x,y,10,10);

axis([0,11,0,11]);

hold on;

text(x(DoveWinning),y(DoveWinning),'Dove');
text(x(HenWinning),y(HenWinning),'Hen');
text(x(DuckWinning),y(DuckWinning),'Duck');
text(x(GooseWinning),y(GooseWinning),'Goose');
text(x(OwlWinning),y(OwlWinning),'Owl');
text(x(HawkWinning),y(HawkWinning),'Hawk');
text(x(EagleWinning),y(EagleWinning),'Eagle');
text(x(FoxWinning),y(FoxWinning),'Fox');
text(x(DogWinning),y(DogWinning),'Dog');
text(x(WolfWinning),y(WolfWinning),'Wolf');
text(x(CatWinning),y(CatWinning),'Cat');
text(x(TigerWinning),y(TigerWinning),'Tiger');
text(x(LionWinning),y(LionWinning),'Lion');
text(x(HorseWinning),y(HorseWinning),'Horse');
text(x(ZebraWinning),y(ZebraWinning),'Zebra');
text(x(CowWinning),y(CowWinning),'Cow');
