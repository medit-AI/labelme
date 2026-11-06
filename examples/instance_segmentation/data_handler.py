from pathlib import Path
from datetime import datetime
from statistics import median
from typing import Iterable, Generator, Dict, List, TypedDict, Optional


class TimeStats(TypedDict):
    count: int
    first_time: Optional[str]  # 최초 수정 시각
    last_time: Optional[str]   # 마지막 수정 시각
    median_time: Optional[str] # 중간값 시각 (Median)
    mean_time: Optional[str]   # 평균 시각 (Mean)
    time_span_sec: float       # 시간 범위 (초 단위)


def parse_timestamp(time_str: str, time_format: str = "%Y-%m-%d %H:%M:%S") -> float:
    """strftime 포맷 문자열을 POSIX 타임스탬프로 변환"""
    return datetime.strptime(time_str, time_format).timestamp()


def format_timestamp(ts: float, time_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """POSIX 타임스탬프를 strftime 포맷 문자열로 변환"""
    return datetime.fromtimestamp(ts).strftime(time_format)


def filter_files_by_ext(
    root_dir: str | Path, 
    extensions: Iterable[str]
) -> Generator[Path, None, None]:
    """지정한 디렉터리를 Deep search하며 확장자 필터링된 파일만 제너레이터로 반환"""
    root = Path(root_dir)
    ext_set = {ext.lower() if ext.startswith('.') else f".{ext.lower()}" for ext in extensions}
    
    return (
        file for file in root.rglob("*")
        if file.is_file() and file.suffix.lower() in ext_set
    )


def partition_files_by_mtime(
    files: Iterable[Path], 
    target_timestamp: float
) -> Dict[str, List[Path]]:
    """파일 목록을 기준 타임스탬프 이전/이후로 분류"""
    result: Dict[str, List[Path]] = {"before": [], "after": []}
    
    for file in files:
        key = "after" if file.stat().st_mtime >= target_timestamp else "before"
        result[key].append(file)
        
    return result


def get_mtime_stats(
    files: Iterable[Path], 
    time_format: str = "%Y-%m-%d %H:%M:%S"
) -> TimeStats:
    """
    [NEW] 파일 리스트의 타임스탬프 통계 산출 (최초, 마지막, 중간시각)
    """
    mtimes = [f.stat().st_mtime for f in files]
    
    if not mtimes:
        return {
            "count": 0,
            "first_time": None,
            "last_time": None,
            "median_time": None,
            "mean_time": None,
            "time_span_sec": 0.0
        }

    min_ts = min(mtimes)
    max_ts = max(mtimes)
    med_ts = median(mtimes)
    mean_ts = sum(mtimes) / len(mtimes)

    return {
        "count": len(mtimes),
        "first_time": format_timestamp(min_ts, time_format),
        "last_time": format_timestamp(max_ts, time_format),
        "median_time": format_timestamp(med_ts, time_format),
        "mean_time": format_timestamp(mean_ts, time_format),
        "time_span_sec": max_ts - min_ts
    }


def print_summary(
    classified_data: Dict[str, List[Path]], 
    target_time_str: str, 
    root_dir: str | Path,
    time_format: str = "%Y-%m-%d %H:%M:%S"
) -> None:
    """통계 데이터를 포함한 요약 출력"""
    before_files = classified_data["before"]
    after_files = classified_data["after"]
    all_files = before_files + after_files

    print(f"📁 Root Path : {Path(root_dir).resolve()}")
    print(f"⏳ 기준 시간  : {target_time_str}")
    print(f"📊 개수 요약  : 이전({len(before_files)}개) | 이후({len(after_files)}개) | Total: {len(all_files)}개\n")

    # 전체 파일 시각 통계
    overall_stats = get_mtime_stats(all_files, time_format)
    if overall_stats["count"] > 0:
        print("📈 [전체 파일 시간 통계]")
        print(f"  • 최초 수정 (First)  : {overall_stats['first_time']}")
        print(f"  • 중간 수정 (Median) : {overall_stats['median_time']}")
        print(f"  • 평균 수정 (Mean)   : {overall_stats['mean_time']}")
        print(f"  • 마지막 수정 (Last)  : {overall_stats['last_time']}")
        print(f"  • 시간 범위 (Span)   : {overall_stats['time_span_sec'] / 3600:.2f} 시간")


# ==========================================
# 통합 실행 함수
# ==========================================
def classify_files_with_stats(
    target_dir: str | Path,
    extensions: Iterable[str],
    target_time: str,
    time_format: str = "%Y-%m-%d %H:%M:%S",
    verbose: bool = True
) -> Dict[str, Dict]:
    
    target_ts = parse_timestamp(target_time, time_format)
    matched_files = list(filter_files_by_ext(target_dir, extensions)) # 리스트 변환 (통계 함수 재사용용)
    classified = partition_files_by_mtime(matched_files, target_ts)
    
    if verbose:
        print_summary(classified, target_time, target_dir, time_format)

    return {
        "files": classified,
        "stats_all": get_mtime_stats(matched_files, time_format),
        "stats_before": get_mtime_stats(classified["before"], time_format),
        "stats_after": get_mtime_stats(classified["after"], time_format),
    }



# 📁 Root Path : E:\dataset\Medit_AI Dropbox\Medit_AI Dropbox\Medit_AI\xray_panoramic\seoul_univ
# ⏳ 기준 시간  : 2026-10-26 16:30:00
# 📊 개수 요약  : 이전(2827개) | 이후(511개) | Total: 3338개

# 📈 [전체 파일 시간 통계]
#   • 최초 수정 (First)  : 2026-10-22 16:32:15
#   • 중간 수정 (Median) : 2026-10-22 16:45:46
#   • 평균 수정 (Mean)   : 2026-10-23 23:31:26
#   • 마지막 수정 (Last)  : 2026-11-05 17:58:08
#   • 시간 범위 (Span)   : 337.43 시간



# 📁 Root Path : E:\dataset\Medit_AI Dropbox\Medit_AI Dropbox\Medit_AI\xray_panoramic\cbct_ios_dcm
# ⏳ 기준 시간  : 2026-10-26 16:30:00
# 📊 개수 요약  : 이전(418개) | 이후(368개) | Total: 786개

# 📈 [전체 파일 시간 통계]
#   • 최초 수정 (First)  : 2026-08-11 13:59:36
#   • 중간 수정 (Median) : 2026-10-22 10:57:44
#   • 평균 수정 (Mean)   : 2026-10-25 05:21:14
#   • 마지막 수정 (Last)  : 2026-11-06 12:27:22
#   • 시간 범위 (Span)   : 2086.46 시간
  
if __name__ == "__main__":
    res = classify_files_with_stats(
        # target_dir="E:/dataset/Medit_AI Dropbox/Medit_AI Dropbox/Medit_AI/xray_panoramic/seoul_univ",
        target_dir='E:/dataset/Medit_AI Dropbox/Medit_AI Dropbox/Medit_AI/xray_panoramic/cbct_ios_dcm',
        extensions=[ "json"],
        target_time="2026-10-26 16:30:00"
    )
    
    

# if __name__ == "__main__":
#     # 다른 스크립트에서 단독 함수만 가져다 쓸 수도 있고, 전체 조율 함수를 부를 수도 있음
#     res = classify_files(
#         target_dir="E:/dataset/Medit_AI Dropbox/Medit_AI Dropbox/Medit_AI/xray_panoramic/seoul_univ",
#         extensions=["json"],
#         target_time="2026-10-26 00:00:00"
#     )