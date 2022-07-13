#include "Utils.h"

enum BlockMatcherMethod {
    USE_BM,
    USE_SGBM
};

class BlockMatcher {
public:
    BlockMatcher(ImagePair imgPair);
    void performBlockMatching(int wSize, int maxDisp, BlockMatcherMethod bm, bool enableWLS);
private:
    ImagePair m_imgPair;
};